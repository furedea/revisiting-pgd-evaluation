%%
%% 先進的なLaTeX環境をサポートする
%% 九州大学 電気情報工学科向け卒論・修論スタイルファイル
%% Created by Reo Kawasoe
%% 2019/01/28
%%
%
%%% How to use ～使用法～
%
% \documentclass[autodetect-engine, dvipdfmx-if-dvi, ja=standard, a4paper, 12pt]{qu_graduate}
% \usepackage{graphicx}
%
% 【必要なパッケージをここに記載】
%
% \title{【タイトル】}
% \author{【名前】}
% \date{【年月】}
% \major{電気情報工学科}
% \department{九州大学 工学部 電気情報工学科}
%
% \begin{document}
%
% \frontmatter
%
% \maketitle
% \tableofcontents
%
% \mainmatter
%
% 【ここに論文本体を書く or include】
%
% \end{document}
%
%
%%% 古いClassファイルとの整合性を保つため，以下の項目が使える
%
% \preface{まえがき}
% \acknowledgment【謝辞】
% \begin{thebibliography}
% \bibitem
% \end{thebibliography}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{qu_graduate}[2019/01/28]
\LoadClassWithOptions{bxjsreport}

%
% 余白調整
%
\setpagelayout{left=25truemm, right=25truemm, top=30truemm, bottom=30truemm}

%
% タイトル項目定義
%
\def\major{\def\@major}
\def\department{\def\@department}
\title{タイトル}
\author{名前}
\major{電気情報工学科}
\department{九州大学工学部電気情報工学科}

%
% ヘッダ，フッタ定義
%
\usepackage{fancyhdr}
\usepackage{etoolbox}

\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\gt 第\ \normalfont\thechapter\ \gt 章~#1}{}}

\fancypagestyle{plain}{%
    \lhead{} % 左ヘッダに出力する内容。空白でも可。
    \chead{} % 中央ヘッダに出力する内容。空白でも可。
    \rhead{\thepage} % 右ヘッダに出力する内容。空白でも可。
    \def\headrulewidth{0pt} % ヘッダと本文領域の区切り線の幅。いらないなら0pt
    \lfoot{} % 左フッタに出力する内容。空白でも可。
    \cfoot{} % 中央フッタに出力する内容。空白でも可。
    \rfoot{} % 右フッタに出力する内容。空白でも可。\thepageはページ番号を出力するコマンド
    \def\footrulewidth{0pt} % フッタと本文領域の区切り線の幅。いらないなら0pt
}

\fancypagestyle{main}{%
    \lhead{\leftmark}
    \chead{}
    \rhead{\thepage}
    \def\headrulewidth{0.8pt}
    \lfoot{}
    \cfoot{}
    \rfoot{\@department}
    \def\footrulewidth{0.8pt}
}

\appto\frontmatter{\pagestyle{plain}}
\appto\mainmatter{\pagestyle{main}}

%
% Chapter最初のページにもヘッダ，フッタを適用
%
\makeatletter
\let\stdchapter\chapter
\renewcommand*\chapter{%
    \@ifstar{\starchapter}{\@dblarg\nostarchapter}%
}
\newcommand*\starchapter[1]{\stdchapter*{#1}\markboth{#1}{}\thispagestyle{fancy}}
\def\nostarchapter[#1]#2{\stdchapter[{#1}]{#2}\thispagestyle{fancy}}
\makeatother

%
% タイトルページ再定義
%
\makeatletter
\renewcommand{\maketitle}{%
    \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let\footnote\thanks
    \null\vfil
    \vskip 60\p@
    \begin{center}%
        {\huge \gt \@title \par}%
        \vskip 22em%
        {\LARGE \gt
            \lineskip .75em
            \begin{tabular}[t]{c}%
                \@author
            \end{tabular}\par}%
        \vskip 0.5em%
        {\LARGE \gt \@date \par}%
        \vskip 5em%
        {\LARGE \gt \@major \par}%
    \end{center}%
    \par
    \@thanks\vfil\null
    \end{titlepage}%
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\title\relax
    \global\let\author\relax
    \global\let\date\relax
    \global\let\and\relax
    \bxjs@annihilate@subtitle
}%
\makeatother

%
% 古いgraduate.clsとの整合性確保
%
\newcommand{\preface}[1]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}}
\newcommand{\acknowledgment}{\chapter*{謝辞}\addcontentsline{toc}{chapter}{謝辞}}
\renewcommand{\refname}{参考文献}
\renewcommand{\bibname}{参考文献}

\makeatletter
\def\@makechapterhead#1{%
    \vspace*{2.7\Cvs}% 欧文は50pt
    {\parindent \z@ \raggedright \normalfont
        \ifnum \c@secnumdepth >\m@ne
        \if@mainmatter
            \LARGE\headfont \@chapapp\thechapter\@chappos
            \par\nobreak
            \vskip \Cvs % 欧文は20pt
        \fi
        \fi
        \interlinepenalty\@M
        \LARGE \headfont #1\par\nobreak
        \vskip 1.8\Cvs}} % 欧文は40pt

\def\@makeschapterhead#1{%
    \vspace*{2.7\Cvs}% 欧文は50pt
    {\parindent \z@ \raggedright
        \normalfont
        \interlinepenalty\@M
        \LARGE \headfont #1\par\nobreak
        \vskip 1.8\Cvs}} % 欧文は40pt
\makeatother
