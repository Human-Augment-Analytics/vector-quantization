# Documentation Index

This directory contains comprehensive documentation for the HAAG Vector Quantization benchmarking framework.

---

## User Guides

Start here if you're using the tool:

- **[USAGE.md](USAGE.md)** - Complete CLI reference with examples
- **[SLURM_GUIDE.md](SLURM_GUIDE.md)** - Running on Georgia Tech ICE cluster
- **[SLURM_SUPPORT_PLAN.md](SLURM_SUPPORT_PLAN.md)** - Production readiness status & roadmap

---

## Developer Guides

Read these if you're contributing or extending the tool:

- **[ADDING_NEW_METHODS.md](ADDING_NEW_METHODS.md)** - How to implement new quantization methods
- **[METRICS_GUIDE.md](METRICS_GUIDE.md)** - Understanding and interpreting metrics
- **[PARAMETER_SWEEP_GUIDE.md](PARAMETER_SWEEP_GUIDE.md)** - Deep dive into parameter sweeps
- **[DESIGN_AUDIT.md](DESIGN_AUDIT.md)** - Code quality audit & recommendations

---

## Implementation Summaries

Detailed records of major changes:

- **[IS-20251014-production-readiness.md](implementation-summaries/IS-20251014-production-readiness.md)** - SLURM/ICE production readiness (Oct 14, 2025)
- **[IS-20251003.md](implementation-summaries/IS-20251003.md)** - Metrics & sweep improvements (Oct 3, 2025)

---

## Quick Navigation

### I want to...

**...run my first benchmark**
→ Start with main [README.md](../README.md) quick start, then [USAGE.md](USAGE.md)

**...run benchmarks on ICE cluster**
→ [SLURM_GUIDE.md](SLURM_GUIDE.md)

**...add a new quantization method**
→ [ADDING_NEW_METHODS.md](ADDING_NEW_METHODS.md)

**...understand what metrics mean**
→ [METRICS_GUIDE.md](METRICS_GUIDE.md)

**...know what's left to implement**
→ [SLURM_SUPPORT_PLAN.md](SLURM_SUPPORT_PLAN.md)

**...see what changed recently**
→ [Implementation Summaries](implementation-summaries/)

**...understand code quality**
→ [DESIGN_AUDIT.md](DESIGN_AUDIT.md)

---

## Documentation Standards

### For Implementation Summaries

When adding new implementation summaries:

1. Use filename format: `IS-YYYYMMDD-short-description.md`
2. Include these sections:
   - Overview & Motivation
   - Implementation Details
   - Technical Decisions
   - Testing
   - Known Limitations
   - Future Work
   - Metrics (lines of code, time investment)

3. See [IS-20251014-production-readiness.md](implementation-summaries/IS-20251014-production-readiness.md) as template

### For User Guides

- Start with "why" before "how"
- Include examples for every command
- Add troubleshooting sections
- Use tables for comparisons
- Link to related documentation

### For Developer Guides

- Explain design decisions
- Show before/after code examples
- Include testing instructions
- Reference implementation summaries
- Keep up to date with code changes

---

## Contributing to Documentation

### When to Update Docs

Update documentation when you:

- Add new CLI commands
- Change existing command behavior
- Add new features
- Fix bugs that users might encounter
- Make architectural changes
- Complete major implementations

### Documentation Checklist

Before considering a feature "done":

- [ ] Updated relevant user guide
- [ ] Updated relevant developer guide
- [ ] Created implementation summary (if major change)
- [ ] Updated main README if needed
- [ ] Added examples
- [ ] Tested all code examples work

---

## Documentation Status

| Document | Status | Last Updated | Needs Update |
|----------|--------|--------------|--------------|
| USAGE.md | ✅ Current | 2025-10-14 | No |
| SLURM_GUIDE.md | ✅ Current | 2025-10-14 | No |
| SLURM_SUPPORT_PLAN.md | ✅ Current | 2025-10-14 | No |
| ADDING_NEW_METHODS.md | ✅ Current | 2025-10-14 | No |
| METRICS_GUIDE.md | ✅ Current | 2025-10-03 | No |
| PARAMETER_SWEEP_GUIDE.md | ✅ Current | 2025-10-03 | No |
| DESIGN_AUDIT.md | ✅ Current | 2025-10-14 | No |
| IS-20251014 | ✅ Complete | 2025-10-14 | N/A |
| IS-20251003 | ✅ Complete | 2025-10-03 | N/A |

---

## External Resources

### HAAG Project

- [Enrollment Roster](https://gtvault.sharepoint.com/:x:/s/HAAG/EbRWUBbmh3pPpGuh9HF34DgBPnJQEdtMQoBTtANXCxOg9Q?e=B8ykCV)
- [Weekly Report Template](https://gtvault.sharepoint.com/:w:/s/HAAG/EcKDOtAbNKZEr3KrZDfRlZ4BD_IMA-4hTSc7ll52J6_79A)
- [Vector Quantization Project Doc](https://gtvault-my.sharepoint.com/:w:/r/personal/smussmann3_gatech_edu/_layouts/15/Doc.aspx?sourcedoc=%7B805CAAA2-48BB-42CD-A20D-C04F2DA3CA41%7D)

### Technical References

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [ICE Cluster Guide](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042088)
- [Typer Documentation](https://typer.tiangolo.com/)

### Research Papers

- Product Quantization for Nearest Neighbor Search (Jégou et al., 2011)
- [Add key papers as they're referenced]

---

## Getting Help

**Questions about usage:**
→ Check [USAGE.md](USAGE.md) first, then ask in `#vector-quantization` Slack

**Questions about implementation:**
→ Check developer guides, then post in Slack with `@dlevyph`

**Found a bug:**
→ Check [DESIGN_AUDIT.md](DESIGN_AUDIT.md) known issues, then report in Slack

**Feature requests:**
→ Check [SLURM_SUPPORT_PLAN.md](SLURM_SUPPORT_PLAN.md) future work, discuss in weekly meetings

---

*Documentation index last updated: October 14, 2025*
