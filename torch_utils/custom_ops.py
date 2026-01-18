# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid

import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        # 标准 C 盘路径
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
        # D 盘标准路径
        'D:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Program Files*/Microsoft Visual Studio */vc/bin',
        # D 盘根目录路径（你的实际安装路径）
        'D:/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Microsoft VisualStudio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Microsoft VisualStudio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'D:/Microsoft VisualStudio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    name = torch.cuda.get_device_name().lower()
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Auto-detect and set CUDA_HOME FIRST, before any compilation attempts
        if 'CUDA_HOME' not in os.environ and 'CUDA_PATH' not in os.environ:
            cuda_patterns = [
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*',
                'D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*',
                'C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA/*',
                'D:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA/*',
            ]
            # Try to find CUDA installation
            for pattern in cuda_patterns:
                matches = sorted(glob.glob(pattern))
                if len(matches):
                    # Use the latest version
                    cuda_home = matches[-1]
                    if os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc.exe')):
                        os.environ['CUDA_HOME'] = cuda_home
                        os.environ['CUDA_PATH'] = cuda_home
                        break
            # If still not found, try to use conda's cudatoolkit
            if 'CUDA_HOME' not in os.environ:
                conda_env = os.environ.get('CONDA_PREFIX', '')
                if conda_env:
                    library_path = os.path.join(conda_env, 'Library')
                    # Check for CUDA runtime libraries
                    cuda_dlls = glob.glob(os.path.join(library_path, 'bin', 'cudart*.dll'))
                    if cuda_dlls:
                        # Create lib/x64 directory if it doesn't exist (PyTorch expects this)
                        lib_x64_path = os.path.join(library_path, 'lib', 'x64')
                        if not os.path.exists(lib_x64_path):
                            os.makedirs(lib_x64_path, exist_ok=True)
                            # Copy CUDA DLLs to lib/x64 for PyTorch to find
                            for dll in cuda_dlls:
                                dll_name = os.path.basename(dll)
                                target_dll = os.path.join(lib_x64_path, dll_name)
                                if not os.path.exists(target_dll):
                                    try:
                                        shutil.copy2(dll, target_dll)
                                    except:
                                        pass  # If copy fails, continue anyway
                        # Set CUDA_HOME to Library directory
                        os.environ['CUDA_HOME'] = library_path
                        os.environ['CUDA_PATH'] = library_path
        
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Some containers set TORCH_CUDA_ARCH_LIST to a list that can either
        # break the build or unnecessarily restrict what's available to nvcc.
        # Unset it to let nvcc decide based on what's available on the
        # machine.
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        #
        # EDIT: We now do it regardless of TORCH_EXTENSIOS_DIR, in order to work
        # around the *.cu dependency bug in ninja config.
        #
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1: # and ('TORCH_EXTENSIONS_DIR' in os.environ):

            # Compute combined hash digest for all source files.
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # Select cached build directory name.
            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}')

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir) # atomic
                except OSError:
                    # source directory already exists, delete tmpdir and its contents.
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir): raise

            # Compile.
            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)

        # Load.
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
