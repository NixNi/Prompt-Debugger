#!/usr/bin/env bash
# Quick start — run from project root with LMStudio running.
set -euo pipefail
cd "$(dirname "$0")"

python3 main.py \
  --text "The internet began as a military research project in the late 1960s. ARPANET connected the first four universities in 1969, allowing them to share data over phone lines. By the 1980s, TCP/IP became the standard protocol, unifying fragmented networks into a single global system. Tim Berners-Lee invented the World Wide Web in 1989 at CERN, creating HTML and HTTP to let researchers share documents hyperlinked across the network. The first graphical browser, Mosaic, arrived in 1993 and made the web accessible to ordinary people for the first time. Commercial use exploded in the mid-1990s as companies like Amazon and eBay proved that online commerce could work at scale. Broadband replaced dial-up in the 2000s, enabling rich media, streaming video, and the rise of social platforms like Facebook and YouTube. By the 2010s, smartphones put the internet in everyone's pocket, fundamentally changing how humans communicate, work, and consume information." \
  --text2 "The internet transformed global communication by making instant messaging and email the norm. Remote work became viable when cloud computing and video conferencing matured in the 2010s. E-commerce reshaped retail, with physical stores increasingly competing against online giants. Education shifted online through platforms like Coursera and Khan Academy, democratizing access to knowledge. Healthcare adopted telemedicine, allowing doctors to consult patients remotely. The rise of artificial intelligence, powered by vast datasets gathered online, now automates tasks from customer service to medical diagnosis. Cybersecurity emerged as a critical concern as data breaches and ransomware attacks grew more sophisticated. The digital divide remains a pressing issue, with billions still lacking reliable internet access." \
  --point "The first email was sent in 1971 over ARPANET" \
  --point "Blockchain technology emerged from cryptocurrency research" \
  --point "Cloud computing replaced physical server rooms" \
  --point "Social media algorithms optimize for engagement over accuracy" \
  --point "Fiber optic cables carry most intercontinental internet traffic" \
  --point "The Internet of Things connects everyday devices to the network" \
  --model "text-embedding-qwen3-embedding-0.6b" \
  --timeout 240 \
  --output-dir results/example
