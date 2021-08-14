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





#README.md do git original do código

# Emotion detection using deep learning

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

## Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/atulapra/Emotion-detection.git
cd Emotion-detection
```

* Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing) and unzip it inside the `src` folder. This will create the folder `data`.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

* If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:  

```bash
cd src
python emotions.py --mode display
```

* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

![Accuracy plot](../emotionrecognition/imgs/accuracy.png)

## Data Preparation (optional)

* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.

* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## Example Output

![Mutiface](../emotionrecognition/imgs/multiface.png)

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.
