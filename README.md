# Exploration of Genetic Algorithm
This project was an exploration into Genetic Algorithm use in generating images. Given an image and a set number of circles can we approximate the image?

![Mandlebrot Set](images/mandlebrot.png "Mandlebrot Set")

# Method
We utilized a pool of individual circles and performed crossover and mutations for several generations. The best circle was found, then the process was repeated.

This was performed on some simple images to ensure it was working correctly:

<p align="center">
  <img src="images/test.png?raw=true" alt="test"/>
  <img src="images/test0.png?raw=true" alt="test0"/>
</p>


Then we performed tests on more complicated images:

<p align="center">
  <img src="images/mona_lisa.png?raw=true" alt="Mona Lisa"/>
  <img src="images/k.png?raw=true" alt="k"/>
  <img src="images/ironman.png?raw=true" alt="Iron Man"/>
</p>

# Progression & Results
  <p align="center"><img src="results/mona_lisa/mona_lisa.gif?raw=true" alt="Mona Lisa"/></p>
  <p align="center">Mona Lisa</p>  
  <p align="center"><img src="results/k/k.gif?raw=true" alt="k"/></p>
  <p align="center">k</p>  
  <p align="center"><img src="results/ironman/ironman.gif?raw=true" alt="Iron Man"/></p>
  <p align="center">Iron Man</p>  
  <p align="center"><img src="results/mandlebrot/mandlebrot.gif?raw=true" alt="Mandlebrot Set"/></p>
  <p align="center">Mandlebrot Set</p>
