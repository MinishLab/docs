import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://minish.ai',
  redirects: {
    '/packages': '/packages/overview/',
  },
  integrations: [
    sitemap(),
    starlight({
      title: 'Minish',
      components: {
        Header: './src/components/Header.astro',
      },
      description: 'Fast open-source NLP models and packages',
      logo: {
        light: '/logo/minish_logo_lighter.png',
        dark: '/logo/minish_logo_lighter.png',
        replacesTitle: false,
      },
      favicon: '/logo/minish_logo.png',
      customCss: ['./src/styles/custom.css'],
      head: [
        {
          tag: 'script',
          attrs: {
            src: 'https://www.googletagmanager.com/gtag/js?id=G-LQWDNXKF2X',
            async: true,
          },
        },
        {
          tag: 'script',
          content: `window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-LQWDNXKF2X');`,
        },
      ],
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/minishlab' },
        { icon: 'linkedin', label: 'LinkedIn', href: 'https://linkedin.com/company/minish-lab' },
        { icon: 'x.com', label: 'X', href: 'https://x.com/minishlab' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/4BDPR5nmtK' },
      ],
      plugins: [
        starlightBlog({
          title: 'Blog',
          authors: {
            minish: { name: 'Minish Lab', url: 'https://minish.ai' },
          },
        }),
      ],
      sidebar: [
        {
          label: 'Packages',
          items: [
            { label: 'Overview', link: '/packages/overview/' },
            {
              label: 'Model2Vec',
              items: [
                { label: 'Introduction',  link: '/packages/model2vec/introduction/' },
                { label: 'Installation',  link: '/packages/model2vec/installation/' },
                { label: 'Inference',     link: '/packages/model2vec/inference/' },
                { label: 'Distillation',  link: '/packages/model2vec/distillation/' },
                { label: 'Training',      link: '/packages/model2vec/training/' },
                { label: 'Models',        link: '/packages/model2vec/models/' },
                { label: 'Results',       link: '/packages/model2vec/results/' },
                { label: 'Integrations',  link: '/packages/model2vec/integrations/' },
              ],
            },
            {
              label: 'SemHash',
              items: [
                { label: 'Introduction',            link: '/packages/semhash/introduction/' },
                { label: 'Installation',            link: '/packages/semhash/installation/' },
                { label: 'Deduplication',           link: '/packages/semhash/deduplication/' },
                { label: 'Outlier Filtering',       link: '/packages/semhash/outlier-filtering/' },
                { label: 'Representative Sampling', link: '/packages/semhash/representative-sampling/' },
                { label: 'Benchmarks',              link: '/packages/semhash/benchmarks/' },
                { label: 'Custom Encoders',         link: '/packages/semhash/custom-encoders/' },
              ],
            },
            {
              label: 'Vicinity',
              items: [
                { label: 'Introduction',       link: '/packages/vicinity/introduction/' },
                { label: 'Installation',       link: '/packages/vicinity/installation/' },
                { label: 'Usage',              link: '/packages/vicinity/usage/' },
                { label: 'Supported Backends', link: '/packages/vicinity/supported-backends/' },
              ],
            },
            {
              label: 'Semble',
              items: [
                { label: 'Introduction',  link: '/packages/semble/introduction/' },
                { label: 'Installation',  link: '/packages/semble/installation/' },
                { label: 'Usage',         link: '/packages/semble/usage/' },
                { label: 'MCP Server',    link: '/packages/semble/mcp-server/' },
                { label: 'Benchmarks',    link: '/packages/semble/benchmarks/' },
              ],
            },
            {
              label: 'Tokenlearn',
              items: [
                { label: 'Usage', link: '/packages/tokenlearn/usage/' },
              ],
            },
            {
              label: 'Model2Vec-rs',
              items: [
                { label: 'Usage', link: '/packages/model2vec-rs/usage/' },
              ],
            },
          ],
        },
      ],
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
