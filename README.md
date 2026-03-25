# Minish Docs

This repository contains the Minish documentation site, built with Astro and Starlight.

## Development

Install dependencies:

```bash
npm install
```

Start the local development server:

```bash
npm run dev
```

Build the site:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Blog Posts

Add new posts in `src/content/docs/blog/` as `YYYY-MM-DD-slug.mdx`.

Use this frontmatter:

```mdx
---
title: "Post title"
date: 2026-03-25
authors:
  - minish
excerpt: "One short summary sentence for the blog index."
---
```

The blog index is generated automatically from the files in that folder.

## Deployment

The site is deployed with GitHub Pages via GitHub Actions on pushes to `main`.
