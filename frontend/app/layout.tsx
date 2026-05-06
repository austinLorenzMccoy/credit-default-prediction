import type { Metadata } from 'next'
import { Playfair_Display, IBM_Plex_Mono, Lora, DM_Serif_Display } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const playfair = Playfair_Display({ 
  subsets: ["latin"],
  variable: '--font-playfair',
  weight: ['400', '700']
});

const ibmPlexMono = IBM_Plex_Mono({ 
  subsets: ["latin"],
  variable: '--font-ibm-plex-mono',
  weight: ['400', '500', '600']
});

const lora = Lora({ 
  subsets: ["latin"],
  variable: '--font-lora',
  weight: ['400', '500']
});

const dmSerif = DM_Serif_Display({ 
  subsets: ["latin"],
  variable: '--font-dm-serif',
  weight: ['400']
});

export const metadata: Metadata = {
  title: 'CreditIntel - Precision Intelligence',
  description: 'ML-powered credit risk scoring and limit recommendation for financial institutions',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/icon.svg',
        type: 'image/svg+xml',
      },
    ],
    apple: '/apple-icon.png',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${playfair.variable} ${ibmPlexMono.variable} ${lora.variable} ${dmSerif.variable}`}>
      <body className="font-sans antialiased bg-mist">
        {children}
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
