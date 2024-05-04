<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<p align="center">
    <!-- community badges -->
    <a href="https://discord.gg/zZhUEDaQWj"><img src="https://img.shields.io/badge/Join-Discord-blue.svg"/></a>
    <!-- license badge -->
    <a href="https://github.com/danielbob32/ParkingSpace/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a>
</p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/danielbob32/ParkingSpace">

![index](https://github.com/danielbob32/ParkingSpace/assets/120675110/c128eb3a-0221-49b5-b596-d167a89c4740) 
  </a>

<h1 align="center">ParkingSpace Parking Spots Detector</h1>

  <p align="center">
    YOLOv8 based parking detection system using a RTSP camera.
    <br />
    <br />
    <a href="https://github.com/danielbob32/ParkingSpace">View Demo</a>
    ·
    <a href="https://github.com/danielbob32/ParkingSpace/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/danielbob32/ParkingSpace/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#install-and-setup">Install and Setup</a>
      <ul>
        <li><a href="#dependencies">Prerequisites</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#walkthrough">Walkthrough</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#learn-more">Learn More</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

# About The Project

Finding parking spaces has become increasingly challenging in densely populated neighborhoods, where the **number of cars is growing rapidly**. This task not only frustrates drivers but also contributes to traffic congestion as unsuccessful searches add unnecessary delays to journeys. What if there was a solution that could streamline this process?

Introducing ParkingSpace, a Python-based system leveraging YOLOv8 and real-time streaming protocol (RTSP) cameras to revolutionize parking spot detection. Developed as a project during my computer science degree, under the guidance of [Prof. Roi Poranne](https://github.com/Roipo), ParkingSpace aims to alleviate the parking woes encountered in urban areas.

The Challenge
In bustling neighborhoods, the hunt for a vacant parking spot can be maddening. It not only consumes valuable time but also exacerbates traffic congestion as drivers circle blocks in search of elusive spaces. This frustration inspired the creation of ParkingSpace, a solution designed to make urban parking more efficient and less stressful.

The Solution
By harnessing the power of computer vision and YOLOv8, ParkingSpace identifies empty parking spaces in real-time, even within undefined parking areas. This innovative algorithm detects available spots, allowing drivers to make informed decisions about where to park without aimless circling. With ParkingSpace, navigating crowded streets becomes more manageable, reducing overcrowding the street and enhancing the overall urban driving experience on the way home or any other destination.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

# Getting Started

## Install and Setup

This will help you to get started with ParkingSpace if you want to run and experiment with the default street provided.
For more complex changes and setting it to work on other input, please refer to the [references](#references) section.

### Prerequisites

It's highly suggested to run the program on a CUDA compatible NVIDIA video card, although this version manages to use CPU, use at your own risk!
You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
You need  2.5 GB of space to run the demo version.

### Dependencies

Note that ParkingSpace requires `python >= 3.8`.
Before starting, creating a virtual environment is recommended. Vscode guide on how to is [linked-here](https://code.visualstudio.com/docs/python/environments).
Some of the Dependencies will be installed with ultralytics.
```sh
pip install ultralytics
pip install opencv-python
pip install numpy
```

To run with ipython (highly recommended) install:

```sh
pip install jupyter
pip install ipython
```

### Installation

Clone the repo:

```sh
git clone https://github.com/danielbob32/ParkingSpace.git
```

As for now, you should have every thing you need to run the demo with your machine and get to know the system by adjusting the parameters.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- WALKTHROUGH -->

## Walkthrough
On the first run, after you opened a virtual environment and installed all the dependencies the model it-self will be installed, which will take some time depends on your machine.
You will might need to re-run the code to actually start it.

If everything works, about 15 seconds after running the cell you should get the first frame, Congratulations!
Note that this is not showing a live video and set to a specific interval, you can always reduce it to see live footage if your machine can handle it.

![image](https://github.com/danielbob32/ParkingSpace/assets/120675110/710c9c66-7d26-4056-b5d9-6ca8d663cb82)

A window will pop-up and you will able to see the parking spots that are being processed every 15 seconds or so.

![Screenshot 2024-05-02 000605](https://github.com/danielbob32/ParkingSpace/assets/120675110/fc9735be-4666-4770-a94f-cddddd735656)
<br />
<br />
<br />

Note that the model is really heavy to assure the maximum accuracy, if you find the program crashing you can do the following steps :

1. Extend the sampling intervals.

![image](https://github.com/danielbob32/ParkingSpace/assets/120675110/7febc6be-1e54-4b6e-a028-9f947884602e)

2. Reduce the model resolution.

![image](https://github.com/danielbob32/ParkingSpace/assets/120675110/e2d4846d-19ef-479b-a6ac-2e8c3410b38b)

3. Reduce the model accuracy.

![image](https://github.com/danielbob32/ParkingSpace/assets/120675110/f1b3e74e-b69f-4a95-b1be-e659e0611e39)

\_For more options and development, please refer to the [references](#references) section.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [x] Add step-by-step template to init the system anywhere.
- [ ] Make a solid web-app version.
- [ ] Future development:
  - [ ] Self region assignment system.
  - [ ] Plug'n'Play.

See the [open issues](https://github.com/danielbob32/ParkingSpace/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the Apache License, [click here](https://github.com/danielbob32/ParkingSpace/blob/master/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Daniel Bobritski - danielbob32@gmail.com

Project Link: [https://github.com/danielbob32/ParkingSpace](https://github.com/danielbob32/ParkingSpace)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LEARN MORE -->

## Learn More

If you are interested in learning more about how to apply ParkingSpace to your use, develop and more, please check the quicklinks below.
| Section | Description |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ❔ **How To's** |
| [Home Setup]() | Here will be an in-depth guide on how to setup a home system. |
| [How to Record the data]() | Here will be a quick guide on how to acquire your own data. |
| 🖱️ **Developers** |
| [Training the system]() | Here will be a guide on how to train the system on your own data. |
| [Effective Regions]() | Here will be a guide to choosing the right regions. |
| [Contributing]() | Walk-through for how you can start contributing now. |
| 💚 **Community** |
| [Discord](https://discord.gg/zZhUEDaQWj) | Join our community to discuss more. I would love to hear from you and assist! |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATA -->

## Data

The data that has been used is kindly listed bellow.

| Section                                                                                                  | Description                                                                               |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 🖼️ **Raw Data**                                                                                          |
| [Raw Images](https://drive.google.com/drive/folders/19Vj134JEaQX1-8Ek_UGWvElroVuO2vXz?usp=sharing)       | Set of raw images snipped from the live videos.                                           |
| [Augmented Images](https://drive.google.com/drive/folders/1gHNhspoRk9ewnf64yqftENCU64kqY5OP?usp=sharing) | Augmented data set of the images above to enrich the Probability map (save tracking time.) |
| 🖌️ **Processed Data**                                                                                    |
| [Segmented Images](https://drive.google.com/drive/folders/1Te31EDZKZ-XRcGjyaPD2qc1c4jH3Dpk9?usp=sharing) | YOLOv8 segmented images .                                                                 |
| [Binary Masks](https://drive.google.com/drive/folders/145pIsCr6CX0GDmDCKgXocOVQuMTDszI_?usp=sharing)     | Binary masks made out of the segmentation.                                                |
| [Probability Map](https://drive.google.com/file/d/1fwNCc_sKEZyjcrchR3vL8WX9ULql6Wt2/view?usp=sharing)    | Grey-scale probability map constructed out of the binary masks.                            |

## Acknowledgments

- [Ultralytics](https://docs.ultralytics.com/)
- [Roboflow](https://roboflow.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/danielbob32/ParkingSpace.svg?style=for-the-badge
[contributors-url]: https://github.com/danielbob32/ParkingSpace/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/danielbob32/ParkingSpace.svg?style=for-the-badge
[forks-url]: https://github.com/danielbob32/ParkingSpace/network/members
[stars-shield]: https://img.shields.io/github/stars/danielbob32/ParkingSpace.svg?style=for-the-badge
[stars-url]: https://github.com/danielbob32/ParkingSpace/stargazers
[issues-shield]: https://img.shields.io/github/issues/danielbob32/ParkingSpace.svg?style=for-the-badge
[issues-url]: https://github.com/danielbob32/ParkingSpace/issues
[license-shield]: https://img.shields.io/github/license/danielbob32/ParkingSpace.svg?style=for-the-badge
[license-url]: https://github.com/danielbob32/ParkingSpace/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/danielbobritski
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
