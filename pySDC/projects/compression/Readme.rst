Instructions for using libpressio in the Docker container
---------------------------------------------------------

TODOs
-----
 - Streamline the multiplatform business. See, for instance :ref:`here<https://docs.docker.com/build/building/multi-platform/>`
 - Make use of the `ENTRYPOINT` and `CMD` commands in the Dockerfile to possibly automate the installation process
 - Take care of the "layered inheritance" is that needed, I don't think so. This ties in with the above point.

If you haven't done this already, build the container using
 
```
docker build -t libpressio .
```

in this directory. This creates an image with the name 'libpressio'.
Please pay attention to the platform you are using and you intend to run on. If you use this command on an ARM machine and try to use the image in a GitHub action, it will not run because it requires AMD architecture. You can build a platform specific version for GitHub using

```
docker buildx build --platform linux/amd64 -t libpressio:amd64 .
```

If you are on an ARM machine like me, replace `amd64` by `arm64` to build an image for you local machine. Remember to replace the tag with something useful, such as  `-t libpressio:arm64`.
 
Start the image using

```
docker run -v <local_absolute_path_to_pySDC_installation>:/pySDC -ti --rm libpressio
```

the `-v` does a :ref:`"bind_mount"<https://docs.docker.com/storage/bind-mounts/>` to pySDC on your local machine.
You may have to change the tag to the platform specific version.
We want that because it let's us access the same version of pySDC that we have locally inside the container, in particular with all modifications.

While we have installed all dependencies for pySDC already in the Docker container, we need to install pySDC itself.
Before we do anything, we need to load the libpressio module in spack, because it also contains the Python version that we need. Run

```
spack load libpressio
```

We have mounted the local version at `/pySDC/` in the run command, so let's navigate there and run:

```
cd /pySDC; pip install -e .
```

to finally install pySDC. The above steps can also be executed by running `source install_pySDC.sh` in this directory to save some time.
Now you should be good to go: The pySDC in the container is the one you have locally and the Python should have all modules you need.
Otherwise, just install more stuff using pip.
 
Once the container is built, you can keep using it. Next time, you can start from the `docker run ...` command.
Just don't forget to load libpressio because otherwise you may mix up Python versions.

Have fun!
