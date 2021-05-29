import * as tf from '@tensorflow/tfjs'



// Some radial basis function
const RBF = (phi, center, factor) => tf.tidy(() => {
  var additional_phi_dims = phi.shape.slice(0, -1);
  var additional_center_dims = center.shape.slice(0, -1);

  additional_phi_dims.forEach(() => {center = center.expandDims(0)});
  additional_center_dims.forEach(() => {phi = phi.expandDims(-2)});

  return tf.sigmoid(
    tf.sum(
      phi.sub(center).square(),
      -1  // sum over last axis
    ).mul(-10)
  ).mul(factor);

});

// Loss for each task (sum of some RBFs)
const RBFLoss = (phi, center, factor) => tf.tidy(() => {
  var rbfs = RBF(phi, center, factor);
  return tf.sum(rbfs, -1);
});


const task_1 = {
  factor: tf.tensor([-1, -1.5, 2, -1.5]),
  center: tf.tensor([[0.5, 0.7], [0.4, 0.2], [0.6, 0.2], [0.9, 0.3]])
}

export const loss_1 = phi => RBFLoss(phi, task_1.center, task_1.factor);

export const loss_2 = phi => tf.tidy(() => {
  var [x, y] = phi.split(2, -1);
  x = x.mul(12).sub(6);
  y = y.mul(12).sub(6);

  return (x.square().add(y).sub(11)).square().add(
    (x.add(y.square()).sub(7)).square()
  )
})


export const loss_3 = phi => tf.tidy(() => {
  var [x, y] = phi.split(2, -1);
  var x = x.mul(3.5).sub(1.5);
  var y = y.mul(3.5).sub(0.5);
  return tf.scalar(1).sub(x).square().add(y.sub(x.square()).square().mul(100))
});


export const loss_4 = phi => tf.tidy(() => {
  // Beale Function
  var [x, y] = phi.split(2, -1);
  var x = x.mul(9).sub(4.5);
  var y = y.mul(9).sub(4.5);

  var product = x.mul(y)
  var result = tf.scalar(1.5).sub(x).add(product).square();
  product = product.mul(y);

  result = result.add(
    tf.scalar(2.25).sub(x).add(product).square()
  ).add (
    tf.scalar(2.625).sub(x).add(product).mul(y).square()
  )


  return result;
});
