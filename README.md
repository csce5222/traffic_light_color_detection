<!-- ABOUT THE PROJECT -->

## About The Project

The "Traffic Light Color Detection" project is dedicated to the development of a robust system designed to accurately identify and categorize traffic signal colors, including Green, Yellow, and Red. This endeavor encompasses an extensive exploration of diverse color-based image processing models, a comprehensive evaluation of their performance, and the adept handling of challenges related to feature engineering in image data.

At its core, this research project places a significant emphasis on achieving real-time traffic light detection using various objective detection and image processing algorithms. Additionally, the project leverages a diverse array of datasets and color models to facilitate adaptation to different real-world scenarios and ensure the precise categorization of detected colors.


| Type                | Sub Type             | Algorithm                                      |
|---------------------|----------------------|------------------------------------------------|
| Supervised Learning | Image Classification | [CNN](traffic_light_color_detection/) |

### Built With

This section lists all major frameworks/libraries used to bootstrap this project.

* [![Python][Python.org]][Python-url]
* [![Jupyter][Jupyter.org]][Jupyter-url]
* [![Miniconda][Miniconda.com]][Miniconda-url]
* [![Docker][Docker.com]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Following the instructions below should get you up and running and quickly as possible without googling around to run
the code.

### Prerequisites

Below is the list things you need to use the software and how to install them. Note, these instructions assume you are
using a Mac OS. If you are using Windows you will need to go through these instructions yourself and update this READ
for future users.

1. miniconda
   ```sh
   cd /tmp
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
   bash Mambaforge-$(uname)-$(uname -m).sh
   ```

2. Restart new terminal session in order to initiate mini conda environmental setup
   
3. [Docker, including docker-compose](https://docs.docker.com/engine/install/)

### Installation

Below is the list of steps for installing and setting up the app. These instructions do not rely on any external
dependencies or services outside of the prerequisites above.

1. Clone the repo
   ```sh
   git clone git@github.com:csce5222/traffic_light_color_detection.git
   ```
2. Install notebook
   conda env create -f environment.yml
   conda activate traffic_light_color_detection
3. Build Docker Image (Note, you should be in the same dire)
   ```sh
   docker-compose build
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

In order to view or execute the various notebooks run the following command on any of the sub folders in this directory.

Here is an example to launch the Traffic Light Color Detection Notebooks.

```sh
jupyter notebook
```

Once inside the
notebook [use the following link](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html)
on examples of how to use the notebook.

Here is an example to launch docker on the command line.

```sh
docker-compose up -d
docker-compose exec traffic-light-color-detection bash
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

[Kaggle. Traffic Light Detection Dataset](https://www.kaggle.com/datasets/wjybuqi/traffic-light-detection-dataset)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact
[Arun Kandula](mailto:ArunKumarReddyKandula@my.unt.edu)
<br>
[Larry Johnson](mailto:johnson.larry.l@gmail.com)
<br>
[Sima Siami-Namini](mailto:simasiami@gmail.com)
<br>

Project Link: [https://github.com/csce5222/traffic_light_color_detection](https://github.com/csce5222/traffic_light_color_detection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[Jupyter-url]:https://jupyter.org

[Jupyter.org]:https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white

[Python-url]:https://python.org

[Python.org]:https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white

[Miniconda-url]:https://docs.conda.io/

[Miniconda.com]:https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white

[Docker-url]:https://www.docker.com/

[Docker.com]:https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
