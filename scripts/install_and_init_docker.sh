#!/bin/bash

set -xe

sudo update && sudo upgrade -y

# install docker-related package
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# add docker gpg key in apt-key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# register docker repository in apt
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"

# install docker
sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

# version check
docker -v

# command for docker start
sudo systemctl enable docker

# command for check docker is running
service docker status

getent group docker # group에 docker를 추가. 있었으면 출력 X, 없없으면 출력이 나옴
sudo usermod -aG docker $USER #docker group에 현재 유저를 추가
# 변경사항 적용: 그룹 변경 사항을 적용하기 위해, 사용자가 로그아웃하고 다시 로그인해야 합니다. 또는, 시스템을 재부팅하여 변경사항을 적용할 수도 있습니다.

id -nG # TODO: 아직 docker를 수행할 때 sudo를 붙여야 함. 로그아웃 & 로그인 해도 변하지 않음. 확인할 것.

# DLC(AWS가 제공하는 deep learning container)가 위치해있는 registry인 ECR에 로그인하기 위함.
# 주의: AWS 라는 이름으로 로그인 하는 것임. -> 내 credential을 위 AWS 세팅에서 완료한 이후에 진행할 것.
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
