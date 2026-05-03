#!/bin/bash

# Deploy to Vercel
echo "🚀 Deploying CreditLens Frontend to Vercel..."

# Install Vercel CLI if not present
if ! command -v vercel &> /dev/null; then
  echo "Installing Vercel CLI..."
  npm i -g vercel
fi

# Deploy to Vercel
vercel --prod

echo "✅ Frontend deployed to Vercel!"
echo "🌐 Visit your app at: https://creditlens-pred.vercel.app"
