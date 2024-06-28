#

## Documentation TODO: 
- the bluetooth app 

## Setup
1. python 3.10 or 3.11 is required. At the time of writing,
    - 3.12 is not supported by tensorflow 
    - it appearas that conda for python 3.10 or above cannot be installed on raspberry pi
    - Deadsnake PPA does not work on my raspberry pi 
    - Building it from source was required for my case 
2. install opencv
3. install libcamera
4. pip install -r requirement.txt 
5. install the python binding for libcamera

### Install the python binding for libcamera
- if the distro comes with python 3.10 or 3.11, just do `sudo apt install -y python3-libcamera`. And then make it available to the venv 
- otherwise, read this: https://github.com/raspberrypi/pylibcamera
- if the libcamera version from the distro is not supported, you need to build libcamera from source and then continue. see below for more 
- this rpi-libcamera git clone the libcamera repo and checkout the revision number you specify, and the build the python binding. therefore, make sure all the build dependency exists. this includes but not exclusively
    - things in the build essential package
    - the pybind11 version that is compatible with your python version
    - the meson version that is compatible with your python version
    - if anything fails, pip install them. 
        - `pip install ply PyYAML jinja2 pybind11 meson`
- `pip install rpi-libcamera`, and then `sudo ldconfig`
- if nothing works, you need to manually build the python binding. see below. 

### Building libcamera and the python binding for a specific python installation
- I was not familiar with meson or building in general, so I decided to rebuild the entire libcamera with minimal code changes to specify the python version. One may potentially just build the python binding. **This reinstall the libcamera for the system.**
1. download the source code. see the [raspberry pi documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html#build-libcamera-and-rpicam-apps). 
2. specify the python version in `libcamera/src/py/libcamera/meson.build`
    - see this line: `py3_dep = dependency('python-3', required : get_option('pycamera'))`. This get you the system default python. 
    - go to the terminal, do `pkg-config --list-all | grep python`. this shows you the names meson recognises
    - in my case, it's `py3_dep = dependency('python-3.11', required : get_option('pycamera'))`
    - see this line: `destdir = get_option('libdir') / ('python' + py3_dep.version()) / 'site-packages' / 'libcamera'`
    - add `message(destdir)` afterward so you know where it's installed. 
3. the building process uses python. just to be safe, use the python version for which we need the binding
    - see this line: `py_mod.find_installation('python3', modules : py_modules)`
    - make it `py_mod.find_installation('python3.11', modules : py_modules)`
    - i believe the string here is the command python3.11
4. make sure the meson version and other dependency versions are compatible with your python version
    - `pip3.11 install ply PyYAML jinja2 pybind11 meson`
5. follow the docmentation. if you see an error during `sudo ninja -C build install`
    - you might need to `sudo pip3.11 install meson` and retry 
6. go to the terminal, do `sudo ldconfig`
7. after the build and installation finished, i decided to just copy the folder from `destdir` to ,my python site-packages folder. but one could just specify to correct site-packages folder to meson in step 2. 