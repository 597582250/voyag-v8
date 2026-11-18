# Security Policy

## Reporting a Vulnerability

Use the GitHub Security tab to open a private vulnerability report. Do not
open public issues for security vulnerabilities.

Maintainer contact: [@597582250](https://github.com/597582250).

## Safe Use

Voyag V8 is a research and demonstration model. It is not a safety-rated
vehicle controller and must not be the sole decision source for safety-critical
vehicle actions.

Deployments should enforce application-level rules, including:

- Blocking safety-critical actions until independently validated
- Treating `mutex_violations` as a request for clarification
- Not using model confidence as the only safety threshold
- Validating state and utterance inputs from untrusted callers

## Hosted Service

No hosted inference service is provided with this release. Users run the ONNX
graph locally and are responsible for securing their own deployment.

## License

The release is distributed under Apache License 2.0.
