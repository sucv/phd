# Table of Content<a name="Table_of_Content"></a>

+ [What is NSCC](#WHAT)
+ [NSCC VS Normal Linux Server](#DIFF)
+ [Why NSCC](#WHY)
+ [Technical Know-how](#KH)
    + [Singularity Container](#CAN)
        + [Step One, Install Singularity](#ONE)
        + [Step Two, Define the Container](#TWO)
        + [Step Three, Build the Container](#THREE)
    + [Job Submission](#JOB)
        + [Step Four, Define the Job](#FOUR)
        + [Step Five, Upload Everything](#FIVE)
        + [Step Six, Load the Singularity](#SIX)
        + [Step Seven, Submit the Job](#SEVEN)
        + [Step Eight, Track or Debug](#EIGHT)
+ [End Note](#EN)


After finishing this tutorial, the reader should be able to:

1. Understand the general procedure for NSCC usage.
2. Build a Singularity container in local computer.
3. Upload files to personal directory of NSCC.
4. Define a shell script for a job.
5. Submit a job to NSCC.

To achieve the learning outcome, the reader needs:

0. A NSCC account.
1. A local computer with root access.
2. A text editor.
3. A file transferring tool (e.g., WinSCP).
4. A SSH remote console (e.g., Putty).
5. One or two day's struggle.

*This humble tutorial is written by a rookie. The author would like to excuse any of his incomplete or wrong understandings first :) The author thanks Dr. Ravi and Ms. Hang for their valuable information.* 

# What is NSCC<a name="WHAT"></a>
[Return to Table of Content](#Table_of_Content)

National Supercomputing Centre (NSCC) Singapore is a powerful server for high performance computing. An official introduction can be found [here](https://help.nscc.sg/softwarehardware-information/).

NSCC provides powerful (compared to our group server) [hardwares](https://help.nscc.sg/softwarehardware-information/) and abundant [softwares](https://help.nscc.sg/software-list/).

An NTU account in NSCC with an available project code can apply for no more than 8 Nvidia V100 GPUs (only 1 GPU for personal code), each with 16GB graphic memory. *Our group owns a project code.*

# NSCC VS Normal Linux Server<a name="DIFF"></a>
[Return to Table of Content](#Table_of_Content)

A normal Linux server is nothing but a more powerful computer, with stricter access managements and multiple users. You may first create your virtual environment within your home directory, followed by running your code with the console command `python main.py`. If lucky, your program will immediately start. During the running, some logs, figures, and model files are generated. And finally you got the best model parameters or the test results.

NSCC has even more stricter access, module, and job managements. You can barely directly execute any commands (except those simple ones like `cp` `mv` `rm` `cat` `vim` for file transferring and editing), nor can you install anything. To actually run your fancy code, you have to:

1. Define your job as a shell script, including:
	1. Setting the attributes of your *job*, e.g., the project code, the number of CPU and GPU, the time, and the memory, etc.
	2. Assigning an docker image (either public or personal) and any necessary commands for your code.
2. Submit your job to the server. Your job will queue, then run.
3. All the output will generate in the same way, and the console output will be saved in a text file.

# Why NSCC<a name="WHY"></a>
[Return to Table of Content](#Table_of_Content)

For heavy time-consuming tasks, the training may take over one day, let alone the N-fold (5 or 10 or even leave-one-subject-out) cross validation (CV). A humble PhD student in SCSE, NTU may have the following computational resources to train a deep learning model.

+ Personal workstation.
+ School server (e.g., DGX-1).
+ Group server.
+ Google Colab Pro (Thx for Mengjiao's introduction).
+ Some other personally owned fancy servers.

For my experience I would say our group server has the best credits, followed by Google Colab Pro. But for the tasks above, unless you occupy the whole group server or purchase multiple Colab accounts can you train with practical time cost. It would be perfect to separate the N folds to N tasks, each uses one GPU! This is where the NSCC comes into the picture.

It would be perfect to submit N jobs to NSCC, and obtain the result all together :) But in practical we may still need to wisely share tasks with different options.

# Technical Know-how<a name="KH"></a>
[Return to Table of Content](#Table_of_Content)

Before starting, I shall list some useful references.

The overall help page of NSCC is [here](https://help.nscc.sg/user-guide/). Of all the resources in the help page, [A Beginner’s Guide to Running AI Jobs](https://help.nscc.sg/wp-content/uploads/AI-Guide-v1.1-final-1.pdf) provides very comprehensive introduction. [AI System Quick Start Guide](https://help.nscc.sg/wp-content/uploads/AI_System_QuickStart.pdf) provides specific information.

NSCC employs the PBS Pro for job schedulers. [PBS Pro Online resources](http://www.pbsworks.com/SupportGT.aspx?d=PBS-Professional,-Documentation) provides details on every perspective of the PBS Pro.  [NSCC Job Scheduler (PBS Pro) Quick Start Guide](https://help.nscc.sg/pbspro-quickstartguide/) provides succinct information for NSCC PBS Pro. [PBS Pro Reference Sheet](https://help.nscc.sg/wp-content/uploads/2016/08/PBS_Professional_Quick_Reference.pdf) can serve as a cheat sheet.

Now let's get started. 

## Singularity Container<a name="CAN"></a>
[Return to Table of Content](#Table_of_Content)

A container is a virtual machine. It can be a *folder* containing all the common directories and files of a Linux system, or a succinct compressed *file*. A container can be seen as a system-level virtual environment, compared to App-level counterparts like Anaconda and Virtualenv.

Given a Python code with abundant module imports, the container has better portability and reproductivity across platforms. A fully built container requires only the corresponding Apps (Singularity in our case) to run the code, whereas a fully defined Anaconda environment requires not only the App (Anaconda in this case), but also heavy downloading and installation to actuallyl establish the environment, which is restricted in NSCC.

You are strongly recommended to read [Quick Start](https://sylabs.io/guides/3.0/user-guide/quick_start.html), [Build a Container](https://sylabs.io/guides/3.0/user-guide/build_a_container.html), [Definition File](https://sylabs.io/guides/3.0/user-guide/definition_files.html), and [Bind Paths and Mounts](https://sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html#bind-paths-and-mounts) for reference.

#### Step One, Install Singularity<a name="ONE"></a>
[Return to Table of Content](#Table_of_Content)

Install Singularity following the [Installation](https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps). 

I chose Singularity 3.6.4 because it is currently the highest version supported in NSCC (you may check by `module avail` command in your own NSCC console).

#### Step Two, Define the Container<a name="TWO"></a>
[Return to Table of Content](#Table_of_Content)

Write a definition file `my.def`.

My definition file is as follows. The idea is straightforward, find an existing docker image that is the most relevant to your needs. (To find the relevant existing images, [Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated) is a good starting point. Other available hubs are provided in the official definition file page provided above.) then add extra modules specific to your code, and finally build the image using Singularity. A environment section is included to configure environment variables, so that the Python interpretor in the container can satisfy the Lustre file system in NSCC `/scratch/` directory. A test is included for each trial of build. If failed, add more lines for required module then build again.

```Shell
Bootstrap: docker
From: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
Stage: build

%post
        # Rountine for Ubuntu. Make sure to add -y for an apt install command.
        apt update
        apt autoremove

        # Install everything you need in the virtual system. 
        # You should customize the commands below.

        # For opencv and sk-video, so the pip install later will not fail.
        apt install ffmpeg libsm6 libxext6 -y

        # For dlib.
        apt install build-essential cmake -y
        apt install libopenblas-dev liblapack-dev -y
        apt install libx11-dev libgtk-3-dev -y
        pip install dlib
        pip install imutils

        # For mne
        # This docker image originally has Anaconda installed.
        # So I can use "conda install" for mne.
        conda install -c conda-forge mne

        # For some pip install commands.
        pip install sk-video opencv-python
        pip install facenet

%environment
        # Set this variable to false. So that the container will have this boolean
        # variable in FALSE intrinsically.
        # Note, the characters should be all in upper case.
        # The reason to set this variable is that the file system in NSCC
        # disabled the file locking, whereas the Python "with open" command originally 
        # requires the file to be locked.
        # Setting the variable of the container can override this requirement.

        HDF5_USE_FILE_LOCKING='FALSE'
        export HDF5_USE_FILE_LOCKING

%test   
        # Note, this section is optional.
        # To determine whether the installation above satisfied our needs,
        # we can test whether the Python file in our host machine can be 
        # successfully ran after the build. 
        # To do so, we should call 'python' from within 
        # the guest machine, and run the code stored in the host. 
        # Note, to access directory within the guest machine, simply add
        # ${SINGULARITY_ROOTFS} at the beginning of the path, e.g., 
        # ${SINGULARITY_ROOTFS}/home/ will access the home directory in the
        # guest machine.

        python /path/to/your/test.py

# A full official document for definition file is provided in
# https://sylabs.io/guides/3.6/user-guide/definition_files.html.
```

The first two lines tells Singularity the image is from `docker://pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime`. You may find this url from the [Docker Hub page](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated) (by separating the command `docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime`), as shown in Figure 1. The third line is the *Stage* attribute for any virtual machine softwares. In our case it is merely for convention because we would barely use the restoration function after the image was built. The rest lines install everything we need in order to run our code. You will have your own need so it is up to you to write your part. To determine what to install, we use the `%test` section. The `test.py` could be the main file of your python code.

![](https://i.loli.net/2020/12/29/KD5z4QTrsyXlAIq.png "Docker URL")
>Figure 1. Illustration of how to determine the docker url. 

Be ware of the cuda version. The cuda version in docker image should be consistent to the NSCC system. NSCC currently supports cuda10.1, therefore I did not choose the latest version which is for cuda11. 

In my case I chose a Pytorch image. You may also choose to start from a tensorflow, or even a fresh Linux image, then configure and install everything for your need.

#### Step Three, Build the Container<a name="THREE"></a>
[Return to Table of Content](#Table_of_Content)

Build the container (named `my.sif` in this example) by executing:

```Shell
sudo singularity build my.sif my.def
```

Note that the build requires sudo access and should be done in your local computer.

## Job Submission<a name="JOB"></a>
[Return to Table of Content](#Table_of_Content)

In this section we will first define a job, then submit it. The useful command cheat sheet is provided in [PBS Pro Reference Sheet](https://help.nscc.sg/wp-content/uploads/2016/08/PBS_Professional_Quick_Reference.pdf).

#### Step Four, Define the Job<a name="FOUR"></a>
[Return to Table of Content](#Table_of_Content)

Define the job script.

A job script is a shell script in `.pbs` extension. Let us name it as `job.psb`. My job script is as follows. Note that Any lines started with #PBS are not comments, but the job attributes for PBS Pro to read.

A short version without comments.

```Shell
#!/bin/sh
#PBS -l select=1:ncpus=5:ngpus=1
#PBS -l walltime=00:30:00
#PBS -q dgx
#PBS -P 12001577
#PBS -N semaine
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit $?
singularity run --nv \
        --bind ~/scratch/dataset/:/media/ \
        ~/scratch/my.sif python ~/phd-main/main.py \
        -n_fold $n_fold -fold_to_run $fold_to_run > main_log_{$n_fold}_{$fold_to_run}
```

An abundant version with comments.

```Shell
#!/bin/sh

### =============================================================================================================
### ===========Job attribute definition for PBS Pro==============================================================
### =============================================================================================================

### Note that the first line tells the system that this is a shell script. 
### Any lines started with #PBS are not comments,
### but the job attributes for PBS Pro to load.

### Request a single GPU
### The "select=1" specifies the number of nodes
### To ask for a whole node use "PBS -l select=1:ncpus=40:ngpus=8"
### If you request less than 8 GPU then make the ncpus value
###   five times the ngpus value, e.g. select=1:ncpus=5:ngpus=1
#PBS -l select=1:ncpus=5:ngpus=1

### Specify the time required (maximum 24 hours)
### If you pay 5K SGD monthly then you can require 240 hours for a job :)
#PBS -l walltime=00:30:00

### Select the correct queue. For other available queues 
### type "qstat -Q" in your console to see.
### "dgx" is specifically for GPU computing with Nvidia V100 cards.
#PBS -q dgx

### Specify the project code
### e.g. 41000001 was the pilot project code
### Job will not submit unless this is changed.
### The project code, if any, should appear in the welcome information 
### in the console once you logged into your NSCC account.
### If you did not provide an available project code,
### then you can only apply for one GPU.
#PBS -P 41000001

### Specify the job name
### When the job is done, a log file named semaine.o4432523, and
### an error file named semaine.e4432523, with
### the "o" or "e", and job ID in the rear, will be generated. 
### They records the standard outputs and errors during the job execution.
#PBS -N semaine

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### =============================================================================================================
### ===========Start of commands to run==========================================================================
### =============================================================================================================

### Change to directory where the job was submitted
cd "$PBS_O_WORKDIR" || exit $?

### Since our job is to simply run the python code, and 
### we have built a personal container, we could simply
### run the job by one line. The instructions for each segment 
### are provided below.

singularity run --nv \
        --bind ~/scratch/dataset/:/media/ \
        ~/scratch/my.sif python ~/phd-main/main.py \
        -n_fold $n_fold -fold_to_run $fold_to_run > main_log_{$n_fold}_{$fold_to_run}

### Let us call the command above as "the long version".
### A simplified "short version" of the command is
### singularity run --nv my.sif python main.py > main_log

### We start by explaining the short version. (It will fail because
### the path is wrong. But it is easier to understand.)
### There are generally two segments. 

### The first one "singularity run --nv my.sif" means
### to use singularity to run a command from "my.sif" container, with
### nvidia cuda support enabled.

### The second one is the actual command to run.
### "python main.py > main_log" is exactly the same
### as we did in daily PhD struggle, which is, to execute a Python code,
### and log the console output in a text file named "main_log".

### In the long version, the exact paths for my.sif and main.py are specified.
### So that NSCC will find them correctly.
### Also, there are some arguments to feed "main.py". If your "main.py" has no
### input required, then simply remove "-n_fold $n_fold -fold_to_run $fold_to_run".

### In the long version, "--bind ~/scratch/dataset/:/media/" tells Singularity
### to bind "~/scratch/dataset/" in your personal NSCC directory with
### "/media/" inside the container. For me, my dataset is uploaded to "~/scratch/dataset/" in NSCC,
### and in my Python code, the dataset is presumed to be located in "/media/Semaine".
### As the actual folder "Semaine" is in "~/scratch/dataset/" of my NSCC directory,
### doing so let the Python interpretor in the container access the code from "/media/" 
### in the container, as if they are both in the same computer (but actually the code is outside the container) !
### For the official document of bind path, please see https://sylabs.io/guides/3.6/user-guide/bind_paths_and_mounts.html.

### The environment variable HDF5_USE_FILE_LOCKING in the container
### will be set to False. The reason to do so is that the file system in NSCC
### disabled the file locking, so that our container has to be consistent. Or
### the code will raise error when reading a file by "with open" command.

### Some other commands that may be useful are provided below.
### Print the environment variable of NSCC PBS Pro. So 
### that you will know what the goddammit variables in
### https://help.nscc.sg/wp-content/uploads/2016/08/PBS_Professional_Quick_Reference.pdf
### are.
### singularity run path/to/my.sif printenv > log1

### Print the initial graphic card information for this job
### nvidia-smi > log2 
```

#### Step Five, Upload Everything<a name="FIVE"></a>
[Return to Table of Content](#Table_of_Content)

Upload the container file (`my.sif` in our example), your Python code, job script (`job.psb` in our example) and dataset to an appropriate location of your NSCC personal directory (choose between `/scratch` and `/home`). 

By appropriate I mean to refer to the bind path in your job script, and the suggestions in the [AI System Quick Start Guide](https://help.nscc.sg/wp-content/uploads/AI_System_QuickStart.pdf). The latter is quoted below:

>The `/home` file system (home and project directories) is mounted and visible on all login and DGX nodes and inside Docker containers. This file system should be used for storing job scripts, logs and archival of inactive datasets. Active datasets which are being used in calculations should be placed on either the Lustre `/scratch` file system or the local SSD /raid file systems.

Accordingly, I put the Python code and job script in my `~/`, and put the dataset and container in `/scratch`.

The tools for uploading can be varied, either by command line or GUI. I use [WinSCP](https://winscp.net/eng/download.php) in my local computer with WIN10 system. I prefer GUI since the uploading operation can be quite frequent sometimes.

#### Step Six, Load the Singularity<a name="SIX"></a>
[Return to Table of Content](#Table_of_Content)

Load the Singularity module in NSCC by executing:

```Shell
module load singularity/3.6.4
```

You can first execute `module avail` to see all the available versions.


#### Step Seven, Submit the Job<a name="SEVEN"></a>
[Return to Table of Content](#Table_of_Content)

Submit your job by executing:

```Shell
qsub job.psb
```

And if your `job.psb` requires variables, use the `-v` flag like

```Shell
qsub -v n_fold=5,fold_to_run=1 job.pbs
```

So that you can call these two variables as `$n_fold` and `$fold_to_run` in the job script, just as I did in mine. In my case, I would submit 5 jobs, each for one fold:

```Shell
qsub -v n_fold=5,fold_to_run=0 job.pbs
qsub -v n_fold=5,fold_to_run=1 job.pbs
qsub -v n_fold=5,fold_to_run=2 job.pbs
qsub -v n_fold=5,fold_to_run=3 job.pbs
qsub -v n_fold=5,fold_to_run=4 job.pbs
```

#### Step Eight, Track or Debug<a name="EIGHT"></a>
[Return to Table of Content](#Table_of_Content)

Now that the job has been submitted, you will get a job code. You may choose to check the job status, read the console output (if the job is running), or collect your output. They are all saved in places defined by your job script and Python code.

You can check the status by  `qstat`, where "Q" and "R" in the output denote Queue and Run, respectively. Check the [cheat sheet](https://help.nscc.sg/wp-content/uploads/2016/08/PBS_Professional_Quick_Reference.pdf) for more commands.

If the job finished sooner than expected, then it should mostly raise an error. Suppose your job code is `4433221`, and job name is `job`, then you should have a file `job.o4433221` in your current directory. Use `vim job.o4433221` to check the log and debug.

# End Note<a name="EN"></a>
[Return to Table of Content](#Table_of_Content)

In this tutorial, we included everything in the container. So that the actual job to execute is simply one command. If your Python code is not so demanding in modules, maybe an existing docker image in NSCC can already satisfy. In this case, you may directly load an image from NSCC, and maybe `pip install` some modules on the go, and finally run your code. The [AI System QuickStart](https://help.nscc.sg/wp-content/uploads/AI_System_QuickStart.pdf) provides abundant information for this case.

The queue time of a job may varied from seconds to hours.

That's it. Feel free to email me for any suggestions or questions. 












