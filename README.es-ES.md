

## Razonamiento Deductivo para la Resolución de Problemas Matemáticos con Texto 

Este es el repositorio oficial del paper de ACL-2022 "[Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction](https://arxiv.org/abs/2203.10316)".

<img width="1341" alt="Screenshot 2022-11-14 at 4 42 03 PM" src="https://user-contents.github.com/3351187/201614535-b8323ca0-ed31-4bbc-947f-8e5f0b7b6f7d.png">

### Requisitos
* transformers `pip3 install transformers`
* Pytorch > 1.7.1
* paquete accelerate `pip3 install accelerate` (para entrenamiento distribuido)


### Uso

Para reproducir los resultados, simplemente ejecute nuestros scripts en la carpeta `scripts`.

#### Math23k
Por ejemplo, para reproducir los resultados para el conjunto de datos `Math23k` con la configuración train/val/test,
```shell
bash scripts/run_math23k.sh
```
Ejecute lo siguiente para la configuración train/test
```shell
bash scripts/run_math23k_train_test.sh
```

### Resultados Principales
Reproducimos los resultados principales de **Roberta-base-DeductiveReasoner** en la siguiente tabla.

| Conjunto de Datos                  | Exactitud del Valor | 
|:-------------------------|:--------------:|
| Math23k (train/val/test) |      84.3      | 
| Math23k (train/test)     |      86.0      |
| MAWPS (CV de 5 pliegues)        |      92.0      | 
| MathQA (train/val/test)  |      78.6      |
| SVAMP                    |48.9|

Más detalles se pueden encontrar en el Apéndice C de nuestro artículo. 


### Checkpoints
También proporcionamos los checkpoints de **Roberta-base-DeductiveReasoner** que hemos entrenado en los conjuntos de datos Math23k, MathQA y SVAMP.
No proporcionamos los checkpoints del modelo de 5 pliegues debido a limitaciones de espacio.

|             Conjunto de Datos              | Enlace  | 
|:--------------------------------:|---|
 | Math23k (configuración train/dev/test) | [Enlace](https://drive.google.com/file/d/1TAHbdCKar0gqFzOd76LIYMQyI6hPOmL0/view?usp=sharing)  |
|   Math23k (configuración train/test)   | [Enlace](https://huggingface.co/allanjie/math23k_train_test_roberta-base/tree/main)  |
 |              MathQA              | [Enlace](https://drive.google.com/file/d/1hgqSZwMyFearr_RJebL51ROflqwdsZUv/view?usp=sharing) | 
|              SVAMP               | [Enlace](https://drive.google.com/file/d/1ykI_pTPiCrHhgVA1gVN-yZeB-e0-J0TK/view?usp=sharing)  | 

### Conjuntos de Datos

Los datos para Math23k Five-fold no se han cargado en GitHub debido a su tamaño ligeramente mayor; se han subido [aquí](https://drive.google.com/file/d/1oQZUPeIA6TlNySqjcZhTMQA4-onwljTU/view?usp=sharing) en Google Drive. 



### Citación
Si encuentra útil este trabajo, por favor cite nuestro artículo:
```
@inproceedings{jie2022learning,
  title={Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction},
  author={Jie, Zhanming and Li, Jierui and Lu, Wei},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5944--5955},
  year={2022}
}
```
