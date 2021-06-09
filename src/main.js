import './main.scss';

let target


import Teaser from './components/teaser.html';
target = document.querySelector('#teaser');
target.innerHTML = '';
const Teaser_Component = new Teaser({
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
