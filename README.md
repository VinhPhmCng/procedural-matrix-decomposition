<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--
  <a href="https://github.com/VinhPhmCng/gdscript-sections">
	<img src="https://raw.githubusercontent.com/VinhPhmCng/gdscript-sections/master/addons/gdscript_sections/logo.png" alt="Logo">
  </a>
-->

<h2 align="center">Procedural Matrix Decomposition</h3>

  <p align="center">
	Using Numpy and Pylatex
	<br />
	<br />
	<br />
</p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
	<li><a href="#about-the-project">About The Project</a></li>
	<li><a href="#how-to-use">How To Use</a></li>
	<li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

**Phiên bản tiếng Việt**: [nhánh vn-vi](https://github.com/VinhPhmCng/procedural-matrix-decomposition/tree/vn-vi)

A minimum viable product that is part of the development process of [LAFD](https://github.com/VinhPhmCng/LAFD)

A Python program using Numpy and Pylatex to procedurally generate every step of a matrix decomposition process.

The algorithm used is recursive and depends only on basic matrix operations.
Our aim is to introduce the concept of matrix decomposition in a comprehensible fashion.

### Example
**Input**

<img src="https://raw.githubusercontent.com/VinhPhmCng/procedural-matrix-decomposition/main/images/matrix_A.png" alt="matrix_A" width="30%">

**Output**: [example1.pdf](/examples/example1.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- HOW TO USE -->
## How To Use

### Streamlit

The project is deployed on [Streamlit](https://streamlit.io/) for ease of use, thanks to its Cloud sharing service.

Just head to [this link](https://procedural-matrix-decomposition-fuyaf698zxk4emw4uufsfm.streamlit.app/).


### Local use

**Requirements** (It is recommended to use a virtual environemnt such as [Anaconda](https://www.anaconda.com/).)
- [Python](https://www.python.org/) >= 3.11.5
- [Numpy](https://numpy.org/) and [Pylatex](https://jeltef.github.io/PyLaTeX/current/)
- A Latex compiler such as [MiKTeX](https://miktex.org/) or [TeXLive](https://tug.org/texlive/) AND necessary LaTex packages (MiKTeX should automatically show you packages you need to install when you first run the Python program.)

**Then** download the two files [visual.py](/original/visual.py) and [algo.py](/original/algo.py) located in [/original](/original/)

**Finally**, use your favorite code compiler or the command line and run _visual.py_
```shell
python /path/to/visual.py
```

**To decompose a matrix of your choice**, simply modify the Numpy matrix called _matrix\_A_ located in the main function

<img src="https://raw.githubusercontent.com/VinhPhmCng/procedural-matrix-decomposition/main/images/matrix_A.png" alt="matrix_A" width="30%">

in [visual.py](/original/visual.py)

For more information about Numpy's matrices, head to [numpy.matrix](https://numpy.org/doc/stable/reference/generated/numpy.matrix.html)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License
[MIT License](LICENSE) © [VPC](https://github.com/VinhPhmCng)


<p align="right">(<a href="#readme-top">back to top</a>)</p>