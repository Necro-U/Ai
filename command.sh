#!/bin/bash
# https://github.com/Necro-U/github_actions.git

mkdir temp && cd temp
git init
remote_repo=https://Necro-U:${GITHUB_TOKEN}@github.com/Necro-U/Ai.git
git config http.sslVerify false
git config --global user.name "Automated Binder"
git config --global user.email "actions@users.noreply.github.com"

echo initialized
git remote add origin "${remote_repo}"
git show-ref
git branch --verbose

git lfs install

touch selam.txt
echo selam >> selam.txt

echo token: $GITHUB_TOKEN

git add -A 
git commit -m "Automatik Binder push" || exit 0
git push -f "https://Necro-U:${GITHUB_TOKEN}@github.com/Necro-U/Tensorflow.git" master
