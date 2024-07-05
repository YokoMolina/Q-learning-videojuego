% =======================================================
% Clase TIC_FC_EPN.cls provee la estructura general del documento Trabajo de Integración Curricular para las carreras de la Facultad de Ciencias de la EPN

% Diseñado y creado por: Dr. Paul Acevedo Tapia

% =======================================================

% ========== Inicialización clase

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TIC_FC_EPN}[2021/12/30 Trabajo de Integración Curricular]
\LoadClass[12pt,oneside,a4paper]{report}

% ========== Paquetes

% Codificación texto de entrada a Unicode
\RequirePackage[utf8]{inputenc} 

% Idiomas documento: inglés y español
\RequirePackage[english,spanish]{babel}

% Amplía opciones de impresión de caracteres especiales como (< |), copia y pega texto apropiadamente de documentos (DVI/PS/PDF)
\RequirePackage[T1]{fontenc}

% Ajusta fuente de texto del documento
\RequirePackage{bookman} 

% Control gráficos
\RequirePackage{graphicx}

% Ajuste dimensiones página
\RequirePackage[includeheadfoot]{geometry}

% Manejo escritura matemática
\RequirePackage{amsmath,amsfonts,amsthm,amssymb,mathrsfs}

% Manejo hipervínculos en archivo pdf
\RequirePackage{hyperref}

% Ajustes títulos capítulo, sección, etc.
\RequirePackage{titlesec}

% Ajuste colores texto
\RequirePackage{xcolor}

% Ajusta espaciado documento
\RequirePackage{setspace}
\onehalfspacing

% Ajuste texto a mayúsculas, minúsculas
\RequirePackage{textcase}


% ========== Ajustes generales

% Ajustes dimensiones horizontales página
\setlength{\hoffset}{0cm}
\setlength{\textwidth}{15cm}

% Ajustes dimensiones verticales página
\setlength{\voffset}{0cm}
\setlength{\topmargin}{-1.5cm}
\setlength{\headsep}{0.5cm}
\setlength{\headheight}{0.4cm}
\setlength{\textheight}{23.8cm}
\setlength{\footskip}{0.7cm}

% Ajustes hipervínculos
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=green}

% Formato título capítulo
\titleformat
{\chapter} % command
[display] % shape
{\bfseries\Large} % format
{\Huge Capítulo \ \thechapter} % label
{0.2cm} % sep
{
    \rule{\textwidth}{0.35mm}
    \vspace{0.2cm}
    \centering
}
[
\vspace{-0.5cm}%
\rule{\textwidth}{0.35mm}
]

% ========== Portada

% Variables de información

\newcommand{\titulotic}[1]%
    {\newcommand\@titulotic{#1}}
\newcommand{\profesion}[1]%
    {\newcommand\@profesion{#1}}
\newcommand{\estudiante}[1]%
    {\newcommand\@estudiante{#1}}
\newcommand{\correoestudiante}[1]%
    {\newcommand\@correoestudiante{#1}}
\newcommand{\director}[1]%
    {\newcommand\@director{#1}}
\newcommand{\correodirector}[1]%
    {\newcommand\@correodirector{#1}}
\newcommand{\fecha}[1]%
    {\newcommand\@fecha{#1}}


% Contenido portada

\newcommand\portada{%
    \thispagestyle{empty}

    \begin{minipage}{2.7cm}
        \begin{flushleft}
            \includegraphics[scale=0.076]{Imagenes/logoEPNnuevo.png}
        \end{flushleft}
    \end{minipage}
    \hfill
    \begin{minipage}{2.7cm}
        \begin{flushright}
            \includegraphics[scale=0.070]{Imagenes/logoCiencias.png}
        \end{flushright}
    \end{minipage}\vspace{0.5cm}
    \vspace{0.5cm}
    \begin{center}
        {\LARGE\textbf{ESCUELA POLITÉCNICA NACIONAL}}\\[2.5\baselineskip]
        {\Large\textbf{FACULTAD DE CIENCIAS}}\\[2.5\baselineskip]
        {\large\textbf{\MakeTextUppercase{\@titulotic}}}\\
        [2.5\baselineskip]
        {\textbf{TRABAJO DE INTEGRACIÓN CURRICULAR PRESENTADO COMO REQUISITO PARA LA OBTENCIÓN DEL TÍTULO DE \MakeTextUppercase{\@profesion}}}\\[2.5\baselineskip]
        {\textbf{\MakeTextUppercase{\@estudiante}}}\\[0.1\baselineskip]
        {\@correoestudiante}\\[2.5\baselineskip]
        {\textbf{DIRECTOR: \MakeTextUppercase{\@director}}}\\[0.1\baselineskip]
        {\@correodirector}\\[2.5\baselineskip]
        {\textbf{\MakeTextUppercase{\@fecha}}}
    \end{center}
    \vspace*{\fill}
    \newpage
}

% ========== Certificaciones

% Reserva espacio para la firma
\newcommand{\firma}{
    \par\noindent\makebox[6.5cm]{\hrulefill}
}

\newcommand{\certificaciones}{
    \setlength{\parskip}{0.3\baselineskip}
    \setstretch{1.3}
    \vspace*{\fill}
    \begin{center}
        \textbf{CERTIFICACIONES}
    \end{center}

    \vspace{2\baselineskip}
    
    Yo, \MakeTextUppercase{\@estudiante}, declaro que el trabajo de integración curricular aquí descrito es de mi autoría; que no ha sido previamente presentado para ningún grado o calificación profesional; y, que he consultado las referencias bibliográficas que se incluyen en este documento.

        \vspace{2cm}
        \begin{center}
            \firma\\
            \@estudiante
        \end{center}
        
        \vspace{1cm}Certifico que el presente trabajo de integración curricular fue desarrollado por \@estudiante, bajo mi supervisión.
        
        \vspace{2cm}
        \begin{center}
            \firma\\
            \@director\\
            \textbf{DIRECTOR}
        \end{center}

     \vspace*{\fill}
     \newpage
}

% ========== Declaración de autoría
\newcommand{\declaracionautoria}{
     \vspace*{\fill}
     \begin{center}
         \textbf{DECLARACIÓN DE AUTORÍA}
     \end{center}

     \vspace{2\baselineskip}
     
     A través de la presente declaración, afirmamos que el trabajo de integración curricular aquí descrito, así como el(los) producto(s) resultante(s) del mismo, es(son) público(s) y estará(n) a disposición de la comunidad a través del repositorio institucional de la Escuela Politécnica Nacional; sin embargo, la titularidad de los derechos patrimoniales nos corresponde a los autores que hemos contribuido en el desarrollo del presente trabajo; observando para el efecto las disposiciones establecidas por el órgano competente en propiedad intelectual, la normativa interna y demás normas.

    \vspace{1cm}\noindent \@estudiante
    
    \vspace{0.5cm}\noindent \@director
    
    \vspace*{\fill}
    \newpage
}

% ========== Resumen

\newenv{resumen}{
    \vspace*{\stretch{1}}
    \begin{center}
        \textbf{RESUMEN}
    \end{center}
    \vspace{2\baselineskip}
}
{
    \vspace*{\stretch{2}}
    \newpage
}

% ========== Abstract

\newenv{resumenEng}{
    \vspace*{\stretch{1}}
    \begin{center}
        \textbf{ABSTRACT}
    \end{center}
    \vspace{2\baselineskip}
}
{
    \vspace*{\stretch{2}}
    \newpage
}