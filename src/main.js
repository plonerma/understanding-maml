import './main.scss';
import '@fortawesome/fontawesome-free/js/all.js';


let target


import Teaser from './components/teaser.html';
target = document.querySelector('#teaser');
target.innerHTML = '';
const Teaser_Component = new Teaser({
  target: target
});

import FewShotMethods from './components/fewShotMethods.html';
target = document.querySelector('#fewShotMethods');
target.innerHTML = '';
const FewShotMethods_Component = new FewShotMethods({
  target: target
});



import UserOptimizedTheta from './components/userOptimizedTheta.html';
target = document.querySelector('#userOptimizedTheta');
target.innerHTML = '';
const UserOptimizedTheta_Component = new UserOptimizedTheta({
  target: target
});


import MetaGradient from './components/metaGradient.html'
target = document.querySelector('#metaGradient');
target.innerHTML = '';
const MetaGradient_Component = new MetaGradient({
  target: target
})


import ContourExample from './components/contourExample.html'
target = document.querySelector('#contourExample');
target.innerHTML = '';
const ContourExample_Component = new ContourExample({
  target: target
})

import FitSinePretrained from './components/fitSinePretrained.html'
const FitSinePretrained_Component = new FitSinePretrained({
  target: document.querySelector('#fitSinePretrained')
})

/*import FitSineMaml from './components/fitSineMaml.html'
const FitSineMaml_Component = new FitSineMaml({
  target: document.querySelector('#fitSineMaml')
})*/
