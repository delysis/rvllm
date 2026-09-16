# ANE research references

The in-memory private API sequence and MIL operator patterns in
`rvllm-apple-ane-sys/src/in_memory.rs`, `rvllm-apple/src/ane_linear.rs`, and the
experimental attention and dynamic FFN code draw on [maderix/ANE](https://github.com/maderix/ANE),
reviewed at commit `d91c9845c0784dec7753048954fc6d0e8411fe29`.

MIT License

Copyright (c) 2026 maderix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[ANEForge](https://github.com/sbryngelson/ANEForge), reviewed at commit
`caeef8edf13b9ec7a3338826daaa27f99e1663d1`, was also consulted for grouped-query
attention and resident-cache design. Its e5rt execution interface is distinct
from the private in-memory interface used here; support in one must not be
assumed to establish support in the other.
