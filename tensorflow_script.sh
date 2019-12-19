python3 -m pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
python3 -m pip install -U --user keras_applications --no-deps
python3 -m pip install -U --user keras_preprocessing --no-deps
wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
chmod +x bazel-0.26.1-installer-linux-x86_64.sh
./bazel-0.26.1-installer-linux-x86_64.sh --user

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.15
./configure < ../configure_input.txt
bazel build //tensorflow/tools/pip_package:build_pip_package --config=verbs
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
python3 -m pip install --user /tmp/tensorflow_pkg/tensorflow-1.15.0-cp35-cp35m-linux_x86_64.whl
