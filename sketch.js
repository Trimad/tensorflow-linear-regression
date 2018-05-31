let x_arr = [];
let y_arr = [];

let m, b;

const learningRate = 0.33;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(windowWidth, windowHeight);

  m = tf.variable(tf.scalar(0.0));
  m.dispose();
  b = tf.variable(tf.scalar(0.5));
  b.dispose();

}

function draw() {

  background(0);

  for (var i = 0; i < height; i += 5) {
    stroke(51);
    point(width / 2, i);
  }
  for (var i = 0; i < width; i += 5) {
    stroke(51);
    point(i, height / 2);
  }

  tf.tidy(function() {
    if (x_arr.length > 0) {
      const y_tensor = tf.tensor1d(y_arr);
      optimizer.minimize(() => loss(y_tensor, predict_linear(x_arr)));
    }
  });


  stroke(127, 127, 255);
  strokeWeight(8);

//Draw every point
  for (let i = 0; i < x_arr.length; i++) {
    point(map(x_arr[i], 0, 1, 0, width), map(y_arr[i], 0, 1, 0, height));
  }

  const lineX = [0, 1];
  const ys = tf.tidy(() => predict_linear(lineX));
  let lineY = ys.dataSync();
console.log(predict_linear(lineX));
  //let lineY = tf.tensor1d(ys);
  ys.dispose();

  let v1 = createVector(map(lineX[0], 0, 1, 0, width), map(lineY[0], 0, 1, 0, height));
  let v2 = createVector(map(lineX[1], 0, 1, 0, width), map(lineY[1], 0, 1, 0, height));

  if (v1.y > v2.y) {
    stroke(127, 255, 127);
  } else {
    stroke(255, 127, 127);
  }
  strokeWeight(2);
  line(v1.x, v1.y, v2.x, v2.y);

  if (frameCount % 15 === 0) {
    //console.log(tf.memory().numTensors);

    document.title = "FPS: " + Math.round(frameRate());
  }
}

function loss(labels, predictions_linear) {
  return predictions_linear.sub(labels).square().mean();
  //labels.print();
  //predict_linearions.print();

}

function predict_linear(x) {
  const x_arr = tf.tensor1d(x);
  //y=mx+b
  const y_arr = x_arr.mul(m).add(b);
  return y_arr;
}

function mousePressed() {
  const x = map(mouseX, 0, width, 0, 1);
  const y = map(mouseY, 0, height, 0, 1);
  x_arr.push(x);
  y_arr.push(y);
}
