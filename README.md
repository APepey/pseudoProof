# PseudoProof
https://pseudoproof.streamlit.app  
Using ML models and generative AI to identify fabricated datasets in academic papers  

## Project done @ Le Wagon Bootcamp by:  
Anaïs Pepey, Mariano Rubio and Despoina Kotsopoulou  

Inspired from previous research by Michael S. Bradshaw and Samuel H. Payne  
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260395  

## How PseudoProof works  
- Upload a csv dataset you have suspicions about
- Let the random forest model do the work for about 5 seconds
- Download the results: the original dataset completed with our prediction per row
  - 1: this row has likely been fabricated by a machine
  - 0: this row is likely authentic
