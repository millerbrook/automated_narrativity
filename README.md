# The Path to Automated Narrative Processing

A conference presentation (and potential article) exploring how narrative figures into the drive towards artificial intelligences that bear comparison to human cognition.

## Abstract

How does narrative figure into the drive towards artificial intelligences that bear comparison to human cognition?

There is a robust literature about narrative computation, focusing on the analysis of narrative structures. Large language models can be leveraged to generate narratives. While these two directions automate two high-level human cognitive activities, the processes do not resemble how narrative structures are recognized and created in human thinking.

This presentation describes the function of ongoing narrative processing in human intelligence, followed by consideration of the state of artificial intelligence in approximating that functionality.

## Topics Covered

1. **The Problem of Ongoing Narrative Processing** — Orienting ourselves towards, framing responses to, and retroactively parsing experience
2. **Human Event Processing** — What we know about how humans process events
3. **AI Event Processing** — State of the art in artificial event processing
4. **Research Roadmap** — Directions that might lead to human-like event processing in AI
5. **AGI Considerations** — Contributions to Artificial General Intelligence
6. **Ethics & Usefulness** — Embodiment as precondition; narrative processing as limitation vs. feature

## Building

### Presentation (Beamer)
```bash
pdflatex presentation.tex
bibtex presentation
pdflatex presentation.tex
pdflatex presentation.tex
```

### Article
```bash
pdflatex article.tex
bibtex article
pdflatex article.tex
pdflatex article.tex
```

Or use `latexmk`:
```bash
latexmk -pdf presentation.tex
latexmk -pdf article.tex
```

## Project Structure

```
automated_narrativity/
├── presentation.tex    # Beamer slides
├── article.tex         # Article version
├── references.bib      # Shared bibliography
├── figures/            # Images and diagrams
└── sections/           # Modular content sections
```

## License

[Add your license here]
