Instructions for using libpressio in the Docker container
---------------------------------------------------------

If you haven't done this already, build the container using
 
```
docker build -t libpressio .
```

in this directory. This creates an image with the name 'libpressio'

Start the image using

```
docker run -v <local_absolute_path_to_pySDC_installation>:/pySDC -ti --rm libpressio
```

the `-v` does a :ref:`"bind_mount"<https://docs.docker.com/storage/bind-mounts/>` to pySDC on your local machine.
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
