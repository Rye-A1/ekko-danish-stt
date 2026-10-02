# Security

Report security issues privately to the repository owner rather than opening a
public issue. Never include tokens, private model URLs, customer audio, or raw
transcripts in reports or logs.

Model and native-runtime downloads are pinned to immutable revisions and
verified against SHA-256 values. Treat a checksum mismatch as a hard failure;
do not bypass it.