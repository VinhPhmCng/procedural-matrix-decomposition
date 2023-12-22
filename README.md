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

<h2 align="center">Phân Rã Ma Trận Bằng Đệ Quy</h3>

  <p align="center">
	Với Numpy và Pylatex
	<br />
	<br />
	<br />
</p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Mục Lục</summary>
  <ol>
	<li><a href="#về-project">Về Project</a></li>
	<li><a href="#cách-dùng">Cách Dùng</a></li>
	<li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Về Project

**English version**: [main branch](https://github.com/VinhPhmCng/procedural-matrix-decomposition)

Đây là một sản phẩm khả thi tối thiểu (MVP) - đóng góp cho sự phát triển của [LAFD](https://github.com/VinhPhmCng/LAFD).

Một phần mềm Python dùng Numpy và Pylatex để soạn thảo các bước trong quá trình phân rã một ma trận một cách tự động.

Thuật toán được sử dụng mang tính đệ quy và chỉ sử dụng những phép toán ma trận cơ bản nhất.
Mục tiêu của chúng mình là giới thiệu khái niệm phân rã ma trận một cách dễ hiểu.

### Ví dụ
**Input**

<img src="https://raw.githubusercontent.com/VinhPhmCng/procedural-matrix-decomposition/main/images/matrix_A.png" alt="matrix_A" width="30%">

**Output**: [example1.pdf](/examples/example1.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- HOW TO USE -->
## Cách Dùng

### Dùng Streamlit

Project này được triển khai trên [Streamlit](https://streamlit.io/) để tiện sử dụng, nhờ dịch vụ Cloud sharing.

Truy cập app [ở đây](https://procedural-matrix-decomposition-fuyaf698zxk4emw4uufsfm.streamlit.app/).


### Sử dụng trực tiếp

**Yêu cầu** (Bạn nên sử dụng một môi trường ảo như [Anaconda](https://www.anaconda.com/).)
- [Python](https://www.python.org/) >= 3.11.5
- [Numpy](https://numpy.org/) và [Pylatex](https://jeltef.github.io/PyLaTeX/current/)
- Trình biên dịch LaTex như [MiKTeX](https://miktex.org/) hoặc [TeXLive](https://tug.org/texlive/) VÀ những LaTex packages cần thiết (MiKTeX sẽ báo cho bạn những packages cần cài đặt khi bạn lần đầu chạy chương trình Python này.)

**Sau đó**, tải hai files [write.py](/original/write.py) và [algo.py](/original/algo.py) ở thư mục [/original](/original/).

**Cuối cùng**, sử dụng trình biên dịch yêu thích của bạn, hoặc command line, để chạy __.
```shell
python /path/to/write.py
```

**Để phân rã một ma trận bất kỳ**, bạn chỉ cần thay đổi _matrix\_A_ ở trong hàm main

<img src="https://raw.githubusercontent.com/VinhPhmCng/procedural-matrix-decomposition/main/images/matrix_A.png" alt="matrix_A" width="30%">

trong file [write.py](/original/write.py).

Để biết thêm về ma trận của Numpy, truy cập [numpy.matrix](https://numpy.org/doc/stable/reference/generated/numpy.matrix.html).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License
[MIT License](LICENSE) © [VPC](https://github.com/VinhPhmCng)


<p align="right">(<a href="#readme-top">back to top</a>)</p>