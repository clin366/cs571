Getting Started
=====

## Git Repository

* Login to [Github](https://github.com) (create an account if you do not have one). 
* Create a new repository called `cs571` and make it PRIVATE.
* From the `Settings` menu, add the TAs as collaborators of this repository.
  * Gary Lai: `imgarylai`
* Clone the repository on your local machine:

  ```
  $ git clone https://github.com/your_id/cs571.git
  ```

* Under `cs571`, copy and paste the `hw0` directory (the current directory).
* Add `hw0` to git:

  ```
  $ git add hw0
  ```

* Push your changes to [Github](https://github.com):

  ```
  $ git push origin master
  ```


## Cloud Computing

* Login to the [Amazon Web Services](https://aws.amazon.com) (create an account if you do not have one).
* Go to the [AWS Management Console](https://console.aws.amazon.com) and create an EC2 instance.
  * Machine Image: Ubuntu Server 18.04 LTS.
  * Instance Type: `t2.micro`.
  * You may need to create a key pair during this process. Save the key pair on your local machine and never upload it to the web. You will not be able to access the instance without this key pair so remember where you save it.
* Once the instance is launched, connect to the instance using the secure shell (`ssh`):

  ```
  $ ssh -i "your-key-pair.pem" ubuntu@ec2-3-xxx-xxx-xxx.compute-1.amazonaws.com
  ```

* Make sure you STOP the instance when it is idle; it will keep charging you as long as it runs.



## Docker

* Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/) on the EC2 instance.
* Clone your `cs571` repository:

  ```
  $ git clone https://github.com/your_id/cs571.git
  ```

* Go to the `hw0` directory and build a docker image:

  ```
  $ cd cs571/hw0
  $ docker build -t hw0 .
  ```

* Run the docker image:

  ```
  $ docker run -v /home/ubuntu/cs571/hw0:/mnt:rw -e "IN_FILE=/mnt/dat/hw0.in" -e "OUT_FILE=/mnt/dat/hw0.out" hw0
  ```

* You should have `hw0.out` created under the `cs571/hw0/dat` directory. Add `hw0.out` to git and push it to Github:

  ```
  $ git add dat/hw0.out
  $ git push origin master
  ```

## Submission

Submit the git address of your `cs571` repository in Github: https://canvas.emory.edu/courses/54027/assignments/199146

