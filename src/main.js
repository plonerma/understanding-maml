import './main.scss';
import '@fortawesome/fontawesome-free/js/all.js';


import Teaser from './components/teaser.html';
import FewShotMethods from './components/fewShotMethods.html';
import FewShotVenn from './components/fewShotVenn.html';
import UserOptimizedTheta from './components/userOptimizedTheta.html';
import MetaGradient from './components/metaGradient.html'
import FitSinePretrained from './components/fitSinePretrained.html'
import FitSineMaml from './components/fitSineMaml.html'

const componentMap = {
  '#teaser': Teaser,
  '#fewShotMethods': FewShotMethods,
  '#fewShotVenn': FewShotVenn,
  '#userOptimizedTheta': UserOptimizedTheta,
  '#metaGradient': MetaGradient,
  '#fitSinePretrained': FitSinePretrained,
  '#fitSineMaml': FitSineMaml
}


let element

for (let target in componentMap) {
  element = document.querySelector(target)
  if (element) {
    element.innerHTML = '';
    new componentMap[target]({
      target: element
    })
  } else {
    console.log(`Element ${target} not found.`)
  }
}

// All components monted, reexecute mathjax
MathJax.typesetPromise()
