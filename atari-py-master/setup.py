import glob
import os
import sys
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Library


# Force linker to produce a shared library
class build_ext(_build_ext):
    if sys.platform.startswith('linux'):
        def get_ext_filename(self, fullname):
            import setuptools.command.build_ext
            tmp = setuptools.command.build_ext.libtype
            setuptools.command.build_ext.libtype = 'shared'
            ret = _build_ext.get_ext_filename(self, fullname)
            setuptools.command.build_ext.libtype = tmp
            return ret

    def setup_shlib_compiler(self):
        _build_ext.setup_shlib_compiler(self)
        if sys.platform == 'win32':
            from distutils.ccompiler import CCompiler
            mtd = CCompiler.link_shared_object.__get__(self.shlib_compiler)
            self.shlib_compiler.link_shared_object = mtd
        elif sys.platform.startswith('linux'):
            from functools import partial
            c = self.shlib_compiler
            c.link_shared_object = partial(c.link, c.SHARED_LIBRARY)


def list_files(path):
    for root, dirs, files in os.walk(path):
        for fname in files:
            yield os.path.join(root, fname)

        for dirname in dirs:
            for rpath in list_files(os.path.join(root, dirname)):
                yield rpath


basepath = os.path.normpath(r'atari_py/ale_interface/src')
modules = [os.path.join(basepath, os.path.normpath(path))
           for path in 'common controllers emucore emucore/m6502/src '
                       'emucore/m6502/src/bspf/src environment games '
                       'games/supported external external/TinyMT'.split()]
defines = []
cflags = []
ldflags = []
sources = [os.path.join('atari_py', 'ale_c_wrapper.cpp'),
           os.path.join(basepath, 'ale_interface.cpp')]
includes = ['atari_py', basepath, os.path.join(basepath, 'os_dependent')]
includes += modules

for folder in modules:
    sources += glob.glob(os.path.join(folder, '*.c'))
    sources += glob.glob(os.path.join(folder, '*.c?[xp]'))

if sys.platform.startswith('linux'):
    defines.append(('BSPF_UNIX', None))
    for fname in 'SettingsUNIX.cxx OSystemUNIX.cxx FSNodePOSIX.cxx'.split():
        sources.append(os.path.join(basepath, 'os_dependent', fname))
elif sys.platform == "darwin":
    defines.append(('BSPF_MAC_OSX', None))
    includes.append(
        '/System/Library/Frameworks/vecLib.framework/Versions/Current/Headers')
elif sys.platform == "win32":
    defines.append(('BSPF_WIN32', None))
    for fname in 'SettingsWin32.cxx OSystemWin32.cxx FSNodeWin32.cxx'.split():
        sources.append(os.path.join(basepath, 'os_dependent', fname))
    # disable msvc secure warnings
    defines.append(('_CRT_SECURE_NO_WARNINGS', None))
    cflags += ['/O2', '/GL', '/GF', '/EHs-']
    ldflags += ['/LTCG']


ale_c = Library('ale_c',
                define_macros=defines,
                extra_compile_args=cflags,
                extra_link_args=ldflags,
                sources=sources,
                include_dirs=includes,
                )


setup(name='atari-py',
      version='1.2.2',
      description='Python bindings to Atari games',
      url='https://github.com/openai/atari-py',
      author='OpenAI',
      author_email='info@openai.com',
      license='',
      packages=['atari_py'],
      package_data={'atari_py': ['atari_roms/*']},
      cmdclass={'build_ext': build_ext},
      ext_modules=[ale_c],
      install_requires=['numpy'],
      tests_require=['nose2']
      )
