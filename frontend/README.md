# CreditLens Frontend

A modern, professional React application for credit risk assessment and prediction.

## 🚀 Live Application

**URL**: https://credit-default-prediction.vercel.app/

## 🎯 Features

- **Professional Design**: Modern UI with custom color scheme (copper, sage, ink, mist)
- **Responsive Layout**: Mobile-first design with Tailwind CSS
- **Interactive Dashboard**: Real-time risk assessment and credit limit recommendations
- **TypeScript**: Full type safety and developer experience
- **Component Library**: Built with shadcn/ui components
- **Modern Stack**: Next.js 13+ with App Router

## 🛠 Technology Stack

| Technology | Version | Purpose |
|------------|--------|---------|
| **Framework** | Next.js 13+ | React framework with App Router |
| **Language** | TypeScript 5.8 | Type safety and developer experience |
| **Styling** | Tailwind CSS 4.1 | Utility-first CSS framework |
| **Components** | shadcn/ui | Modern React component library |
| **Icons** | Lucide React | Beautiful icon set |
| **Fonts** | Google Fonts | Playfair Display, Lora, IBM Plex Mono |

## 🎨 Design System

### Color Palette
- **Copper** (#C4622D) - Primary brand color
- **Sage** (#4A7C59) - Success/positive states
- **Ink** (#0D1B2A) - Dark text/backgrounds
- **Mist** (#F7F4F0) - Light backgrounds
- **Amber** (#D4A017) - Accent highlights
- **Rule** (#D9D0C7) - Borders and dividers

### Typography
- **Heading**: Playfair Display (serif, bold)
- **Body**: Lora (serif, readable)
- **Mono**: IBM Plex Mono (code, data)

## 📁 Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── page.tsx           # Landing page
│   ├── layout.tsx          # Root layout
│   └── dashboard/          # Dashboard routes
│       ├── page.tsx         # Main dashboard
│       ├── api-status/       # API status page
│       ├── history/          # Prediction history
│       └── profiling/       # Customer profiling
├── components/              # React components
│   ├── ui/               # shadcn/ui components
│   └── theme-provider.tsx  # Theme context
├── lib/                   # Utility functions
│   └── utils.ts           # Tailwind class utilities
├── styles/                # Global styles
├── public/                # Static assets
├── package.json           # Dependencies
├── next.config.mjs        # Next.js config
├── postcss.config.mjs     # PostCSS config
├── tsconfig.json         # TypeScript config
└── vercel.json           # Vercel deployment config
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18.0+
- pnpm (recommended) or npm

### Installation

```bash
cd frontend
pnpm install
```

### Development

```bash
pnpm run dev
```

Application will be available at `http://localhost:3000`

### Build

```bash
pnpm run build
```

### Deployment

The application is automatically deployed to Vercel:
- **Live URL**: https://credit-default-prediction.vercel.app/
- **Build Command**: `pnpm run build`
- **Output Directory**: `.next`

## 🎨 Customization

### Adding New Colors

Colors are defined in `src/index.css`:

```css
:root {
  --copper: #C4622D;
  --sage: #4A7C59;
  --ink: #0D1B2A;
  --mist: #F7F4F0;
}
```

### Using Components

```tsx
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export function MyComponent() {
  return (
    <Button className={cn("bg-copper hover:bg-copper-dk")}>
      Click me
    </Button>
  )
}
```

## 🔧 Configuration

### Environment Variables

Create `.env.local` for development:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### Vercel Environment

Set environment variables in Vercel Dashboard:
- `NEXT_PUBLIC_API_URL`: Production API URL

## 📊 Pages & Routes

### Landing Page (`/`)
- Hero section with value proposition
- Feature highlights and benefits
- Interactive dashboard preview
- API documentation preview
- Call-to-action sections

### Dashboard (`/dashboard`)
- Customer risk assessment form
- Real-time prediction results
- Historical predictions table
- Risk visualization and charts

### API Status (`/dashboard/api-status`)
- Backend health monitoring
- Model status indicators
- Performance metrics

## 🎯 Key Features

### Risk Assessment
- **Default Probability**: ML-powered risk scoring
- **Credit Limit Recommendations**: Dynamic limit suggestions
- **Risk Factors**: Transparent explanation of scores
- **Historical Tracking**: Prediction history and trends

### User Experience
- **Responsive Design**: Works on all devices
- **Smooth Animations**: Professional micro-interactions
- **Loading States**: Skeleton screens and progress indicators
- **Error Handling**: Graceful error recovery

### Developer Experience
- **Type Safety**: Full TypeScript implementation
- **Component Reusability**: Consistent design system
- **Code Organization**: Clean, maintainable structure
- **Performance**: Optimized builds and loading

## 🔍 Debugging

### Common Issues

1. **Styling Not Working**
   - Check Tailwind classes in `src/index.css`
   - Verify `@theme` directive is present
   - Ensure custom colors are defined

2. **API Connection Issues**
   - Verify `NEXT_PUBLIC_API_URL` environment variable
   - Check CORS configuration on backend
   - Ensure backend is running

3. **Build Errors**
   - Clear Next.js cache: `rm -rf .next`
   - Reinstall dependencies: `pnpm install`
   - Check TypeScript types

## 📈 Performance

### Optimization
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Next.js Image component
- **Font Optimization**: Google Fonts with display: swap
- **CSS Purging**: Tailwind removes unused styles

### Metrics
- **Lighthouse Score**: 95+ performance
- **Bundle Size**: Optimized with Next.js
- **Load Time**: < 2 seconds initial load

## 🤝 Contributing

1. Follow existing code patterns
2. Use TypeScript for new components
3. Test on multiple screen sizes
4. Update documentation for new features
5. Use conventional commit messages

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.
