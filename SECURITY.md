# Security

## Static parsing by default

The built-in PyTorch and Keras parsers inspect Python source with the standard-library AST and do not import or execute the target model file.

Do not change the default parser path to execute untrusted model code.

Future runtime tracing backends must be explicit opt-in features and should document that importing or executing a model can run arbitrary Python code.

## Reporting

For security-sensitive issues, avoid publishing exploit details in a public issue until maintainers have had an opportunity to review the report.
