# How to push your part (same steps for everyone)

You need write access to the repo first. Ask Jasmine to add your GitHub account:
repo page, Settings, Collaborators, Add people.

Make sure git uses your own name and the email of your GitHub account:

```
git config --global user.name "Your Name"
git config --global user.email "the-email-on-your-github@example.com"
```

Then, in a terminal:

```
git clone https://github.com/jasminebeshir3-pixel/chest-xray-pneumonia-cnn.git
cd chest-xray-pneumonia-cnn
git checkout main
git pull
git checkout -b <your-branch-name>
```

Copy the files from your folder into the repo, keeping the same paths (for example `src/evaluate.py` goes into the repo's `src` folder). Then:

```
git add <the files listed in your NOTES.md>
git commit -m "<the commit message in your NOTES.md>"
git push -u origin <your-branch-name>
```

Open the repo on GitHub, click "Compare & pull request", and ask the reviewer named in your NOTES.md to review and merge it.

Rules from the spec: commit from your own account, on a feature branch, merged through a pull request that someone else reviews. Don't upload everything in one commit at the end, that loses marks.
