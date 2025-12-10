
# Installing Node & npm (using NVM)

To run the frontend, you must install **Node.js** and **npm**.
The recommended way is through **NVM (Node Version Manager)**.

---

## 1. Install NVM

Run this command in your terminal:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

Then reload your shell:

```bash
source ~/.bashrc   # or ~/.zshrc depending on your shell
```

Verify NVM is installed:

```bash
nvm --version
```

## 2. Install Node.js (version used by this project)

```bash
nvm install 20.19.5
nvm use 20.19.5
```

Check versions:

```bash
node -v     
npm -v      
```

## 3. Install dependencies

```bash
npm install
```

## 4. Run the frontend

```bash
npm run dev
```

Your frontend will start on a local dev server (usually [http://localhost:5173](http://localhost:5173)).
