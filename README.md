# Emotion Recognition

## Técnica de reconhecimento de emoções em faces em câmeras de vigilância


![Badge](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)


<!--ts-->
   * [Sobre](#Sobre)
   
<!--te-->
# Sobre
O ser humano apresenta uma capacidade considerada trivial de reconhecer objetos. Antes mesmo da informação chegar à consciência humana, os módulos sensoriais auditivos e visuais já enriqueceram a percepção com características de alto nível. Por exemplo, ao ver a imagem de uma xícara, não é possível escolher  não reconhecer a xícara. Ao mesmo tempo, a experiência subjetiva do ser humano não é completamente confiável  \citep{geron}.

Baseando-se em estudos relacionados ao córtex visual, o reconhecimento de imagens levou a importante publicação de \citep{lecun} que apresentava a arquitetura LeNet-5 capaz de reconhecer números em cheques manuscritos, utilizando as Redes Neurais Convolucionais (CNNs, em inglês). Além da aplicação em imagens, é possível introduzir as Redes Neurais Convolucionais para reconhecimento de emoções humanas através de vídeos, por exemplo.

A partir das teorias de Darwin de que as emoções humanas seriam universais e partilhadas por todos de uma mesma espécie \citep{darwin}, o psicólogo Dr. Paul Ekman, constatou que todo ser humano gera as mesmas expressões faciais para representar 7 emoções básicas inatas e universais, mapeadas por ele, com a ressalva de que a cultura do lugar pode mascarar emoções negativas ao que se chama "regras de exibição". Entretanto, é possível identificar as seguintes emoções descritas por Paul Ekman: alegria, tristeza, raiva, nojo, medo, surpresa e, posteriormente incluso no conjunto de emoções, o desprezo.\citep{ekman}. Neste artigo, os dados de treinamento e teste serão referentes às seguintes expressões: Raiva, nojo, medo, alegria, neutro, tristeza e surpresa.

O processo de reconhecimento facial é parametrizado a partir de pontos específicos no rosto, tais como olhos, boca e nariz. Neste trabalho, será identificado a face humana através de uma câmera para então criar o reconhecimento das expressões supracitadas utilizando Redes Neurais Convolucionais.

# Status
<h4 align="center"> 
	🚧 Em construção 🚧
</h4>

### Features

- [x] Acesso à base de dados
- [ ] Pré-processamento
- [ ] Treinamento e teste
