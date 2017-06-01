---
title: 学习Latex
layout: post
tags: [Other]
---


```
notepad test.tex
```

```
\documentclass{article}

\begin{document}

Hello,\LaTeX。

\end{document}
```


```
latex test.tex
```

```
This is pdfTeX, Version 3.14159265-2.6-1.40.17 (TeX Live 2016/W32TeX) (preloaded format=latex)
 restricted \write18 enabled.
entering extended mode
(./test.tex
LaTeX2e <2016/03/31>
Babel <3.9r> and hyphenation patterns for 83 language(s) loaded.
(c:/texlive/2016/texmf-dist/tex/latex/base/article.cls
Document Class: article 2014/09/29 v1.4h Standard LaTeX document class
(c:/texlive/2016/texmf-dist/tex/latex/base/size10.clo))
No file test.aux.
[1] (./test.aux) )
(see the transcript file for additional information)
Output written on test.dvi (1 page, 296 bytes).
Transcript written on test.log.
```

#### 将dvi文件转换为pdf文件。
```
dvipdfmx test.dvi
```

```
D:\testlatex>dvipdfmx test.dvi
test.dvi -> test.pdf
[1]
2948 bytes written
```

#### 支持中文的xelatex命令
```
xelatex
```

```
This is XeTeX, Version 3.14159265-2.6-0.99996 (TeX Live 2016/W32TeX) (preloaded format=xelatex)
 restricted \write18 enabled.
entering extended mode
(./test.tex
LaTeX2e <2016/03/31>
Babel <3.9r> and hyphenation patterns for 83 language(s) loaded.
(c:/texlive/2016/texmf-dist/tex/latex/base/article.cls
Document Class: article 2014/09/29 v1.4h Standard LaTeX document class
(c:/texlive/2016/texmf-dist/tex/latex/base/size10.clo)) (./test.aux) [1]
(./test.aux) )
(see the transcript file for additional information)
Output written on test.pdf (1 page).
Transcript written on test.log.
```

#### 做成批处理文件 build.bat
```
latex test.tex
dvipdfmx test.dvi
del *.aux *.dvi *.log
```
buildx.bat
```
xelatex test.tex
del *.aux *.dvi *.log
```

保存test.tex为uft-8编码格式！
```
\documentclass{article}

\usepackage{ctex}

\begin{document}

你好, \LaTeX。

\end{document}
```

#### 查看文档
```
>> texdoc ctex
>> texdoc lshort
```

中文格式 UTF-8
```
\documentclass[UTF8]{ctexart}
\title{My First Document}   
\author{Zhang Tianliang}
\date{\today}

\begin{document}
    \maketitle
    Hello World!
    
    终于可以输入中文了！
    
    Let $f(x)$ be defined by the formula
    $f(x)=3x^2+x-1$.
    
    $$f(x)=3x^2+x-1$$
\end{document}

```
