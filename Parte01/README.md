Parte 1 - SAVI

Miguel Riem Oliveira <mriem@ua.pt>

# Sumário

 - Introdução
 - Apresentação
 - Objetivos
 - Avaliação
 - Introdução ao Linux - O terminal
 - Editores e IDEs
 - Tutoriais de OpenCV

# Pressupostos para a realização dos exercícios
- Ter o Linux instalado (ubuntu 20.04 *altamente recomendado*).
- Instalação do Linux em modo dual boot (*altamente recomendado*, mas também se pode usar máquina virtual).
- Ter o acesso de rede configurado (_wireless_).
*   Consultar as instruções do site dos 
http://www.ua.pt/stic/PageText.aspx?id=15224[sTIC].

# Instalação do Ubuntu

Existem diversas formas de usar o Ubuntu para quem tem outros sistemas
operativos (Windows, MacOS). As mais interessantes são:

  - Uso do live ubuntu: https://tutorials.ubuntu.com/tutorial/try-ubuntu-before-you-install[Try before you install]
  - Instalar uma máquina virtual (`virtualbox` ou outra) e instalar o Linux na máquina virtual.
  - Instalar o Linux em dual-boot com o Windows (**Esta é a forma recomendada. As outras formas serão pouco adequadas em breve**).

As duas primeiras soluções não interferem no disco nem no sistema operativo
existente, mas são mais limitadas em termos de funcionalidades e desempenho. No caso da primeira, todo o trabalho que for feito se perde no fim da sessão se não for copiado para outro local. No caso da máquina virtual, vai ser preciso espaço em disco no ambiente Windows (ou MacOs) para criar a "imagem" do disco onde correrá o Linux em máquina virtual. É uma solução intermédia que funciona relativamente bem, mas como opera sobre o sistema operativo nativo, pode ter limitações de desempenho e ficará dependente da atividade desse sistema operativo (como as atualizações no
Windows).

A **terceira solução** (dual-boot com o sistema operativo nativo) é a mais poderosa porque cada sistema operativo fica no seu próprio espaço e correm separadamente. Porém, é preciso repartir o disco que estaria todo atribuído ao sistema operativo nativo. O Linux oferece esta possibilidade durante a instalação e em geral o processo corre bem, mas há sempre o risco de perda de informação. Por isso, recomenda-se guardar toda a informação importante desenvolvida no sistema operativo original antes de fazer esta forma de instalação.

Mais informações podem ser obtidas por exemplo nos seguintes endereços:

  * https://tutorials.ubuntu.com/
  * https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop

# Apresentação da UC

Ver slides de apresentação da UC.

# Introdução ao Linux e a Shell

Ver o documento  `1-Linux-Breve Introdução`.

# Criação do ambiente e instalação de ferramentas básicas

## Metodologia

Para melhor se desenvolver o trabalho nas aulas, deve-se
seguir uma metodologia de organização de ficheiros em diretórios
por aulas e por exercícios.

Dentro de cada aula, em especial nas primeiras, é também recomendado criar uma subpasta para cada exercício `Ex1`, `Ex2`, etc. Em certas aulas, ou aulas mais avançadas, os diversos exercícios serão feitos por acréscimo sucessivo sobre o código base dos exercícios anteriores; nessa altura serão dadas as instruções nesse sentido.

Os guiões para as aulas estarão a ser continuamente atualizados em:

https://github.com/miguelriemoliveira/savi_23-24

Recomenda-se que, sempre que possível, usem a versão online ou façam o update frequentemente.

## Editor

A ferramenta principal para criar e modificar ficheiros é o editor, muitas
vezes integrado num ambiente de desenvolvimento (IDE). Há inúmeras opções
desde simples editores (`gedit`, `kate`, `kwrite`, etc.) até ambientes de
desenvolvimento muito sofisticados (`codeblocks`, `eclipse`, `vscode`,`pycharm` etc.).

Além das propriedades fundamentais dos editores, hoje em dia são excelentes
_add-ons_ a "automated completion" (preenchimento automático de palavras
e estruturas) , o "syntax highlight" (realce da sintaxe da linguagem),
o "intellissense" (apresentação de todas as opções de preenchimento
automático de campos e estruturas em variáveis, funções, etc.), ou a
inserção automática de fragmentos de código padrão ("code snippets").

O editor com mais tradição por excelência é o "vim" (ou "vi" improuved)
mas a sua utilização eficaz pode requerer anos de prática continuada e
permite todas as facilidades indicadas acima, mas a sua configuração,
por ser praticamente ilimitada, pode-se tornar complexa e, por isso,
contraproducente em utilizadores iniciados.

**Recomenda-se como IDE** o https://code.visualstudio.com/[vscode], que é gratuito.

## OpenCV

O https://opencv.org/[OpenCV] é a biblioteca de referência para processamento de imagem e visão por computador. É open source e gratuita. 

Para além disso tem imensas funcionalidades básicas e avançadas.

Para instalar o OpenCV:

https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html


# Exercícios 

## Exercício 1 - Tutoriais do OpenCV

O OpenCV tem vários tutorials que são uma ajuda valiosa para começar.

https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

Faça os exercícios dos cinco primeiros tutoriais:

 - Introduction to OpenCV
 - Learn how to setup OpenCV-Python on your computer!
 - Gui Features in OpenCV
 - Core Operations
 - Image Processing in OpenCV

