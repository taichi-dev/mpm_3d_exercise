# MPM 3D
The starter code for a 3D material point method simulation demo written in Taichi. This would be a good exercise for you to get used to **continuum simulations**.



https://user-images.githubusercontent.com/2309174/202906099-92c06d35-2e24-4be1-8adb-1354764d74bf.mp4



> To implement your own version, click the "Use this template" button on this page and simply modify the `mpm_3d_starter.py` script.

## Installation
Make sure your `pip` is up-to-date:

```bash
$ pip3 install pip --upgrade
```

Assume you have a Python 3 environment, to install Taichi:

```bash
$ pip3 install -U taichi
```

To run the demo:

```bash
$ python mpm_3d_starter.py
```

## TODOs
The main simulation step `substep` function is intentionally left blank. You shall fill it in with proper code include the P2G, GridOp, and G2P steps. A 2D reference implementation is provided at `mpm99.py`.


## Extra credits are for the extras
There are plenty of room for hacking! We suggest a few of them for you to start with:
- Higher resolution simulations utilizing sparse data structures
- More sophisticated boundary handling and better scenes
- Better particle initialization with arbitaray shapes
- Faster P2G step without floating point atomic operations (integers are fine)
- Higher order time integration methods
- Implicit time integration methods
- Supporting more material models
- Reducing the numerical adhesion/friction/fracture artifacts
- etc.

## Show your work
We encourage you to continue developing on this repo and share your work with our community members. To notify us about your work, make sure you use this repo as a template.
