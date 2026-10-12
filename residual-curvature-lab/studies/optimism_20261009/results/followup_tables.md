| State | r: sgd / soap / ggn / ef / parsec-L | cos_A(m, ḡ) sgd / soap / ggn | EF ceiling max |
| --- | --- | --- | ---: |
| searched_soap-10 | 0.09 / 0.16 / 0.35 / 0.25 / 0.41 | +0.77 / +0.63 / +0.20 | 0.805 |
| searched_soap-25 | 0.14 / 0.28 / 0.59 / 0.28 / 0.87 | +0.35 / +0.30 / +0.18 | 0.744 |
| searched_soap-50 | 0.25 / 0.55 / 1.52 / 0.47 / 3.33 | +0.25 / +0.23 / +0.32 | 0.616 |
| exact_tensor_ggn-10 | 0.13 / 0.27 / 0.58 / 0.38 / 0.87 | +0.67 / +0.49 / +0.32 | 0.774 |
| exact_tensor_ggn-25 | 0.36 / 0.41 / 0.96 / 0.38 / 1.59 | -0.45 / -0.06 / +0.12 | 0.445 |
| exact_tensor_ggn-50 | 0.21 / 0.52 / 1.36 / 0.45 / 2.58 | +0.54 / +0.14 / +0.03 | 0.302 |

| State | Operator | RHS | Held-out 2–4: same / second / corrected / oracle |
| --- | --- | --- | --- |
| searched_soap-10 | sgd | gradient | +0.0359 / +0.0278 / +0.0361 / +0.0362 |
| searched_soap-10 | sgd | momentum | +0.0389 / +0.0321 / — / +0.0393 |
| searched_soap-10 | soap_map | gradient | +0.0416 / +0.0363 / +0.0416 / +0.0417 |
| searched_soap-10 | soap_map | momentum | +0.0487 / +0.0445 / — / +0.0490 |
| searched_soap-10 | exact_ggn | gradient | +0.1324 / +0.1319 / +0.1292 / +0.1325 |
| searched_soap-10 | exact_ggn | momentum | +0.0373 / +0.0374 / — / +0.0377 |
| searched_soap-10 | ef_newton | gradient | +0.1186 / +0.1183 / — / +0.1186 |
| searched_soap-10 | ef_newton | momentum | +0.0291 / +0.0299 / — / +0.0301 |
| searched_soap-10 | parsec_l | gradient | +0.1299 / +0.1316 / +0.1294 / +0.1316 |
| searched_soap-10 | parsec_l | momentum | +0.1321 / +0.1334 / — / +0.1334 |
| searched_soap-25 | sgd | gradient | +0.0152 / +0.0177 / +0.0185 / +0.0202 |
| searched_soap-25 | sgd | momentum | +0.0073 / +0.0143 / — / +0.0171 |
| searched_soap-25 | soap_map | gradient | +0.0193 / +0.0277 / +0.0193 / +0.0302 |
| searched_soap-25 | soap_map | momentum | +0.0177 / +0.0327 / — / +0.0336 |
| searched_soap-25 | exact_ggn | gradient | +0.0803 / +0.0944 / +0.0948 / +0.0950 |
| searched_soap-25 | exact_ggn | momentum | +0.0336 / +0.0385 / — / +0.0385 |
| searched_soap-25 | ef_newton | gradient | +0.1207 / +0.1270 / — / +0.1271 |
| searched_soap-25 | ef_newton | momentum | +0.0286 / +0.0319 / — / +0.0320 |
| searched_soap-25 | parsec_l | gradient | +0.0751 / +0.0909 / +0.0888 / +0.0910 |
| searched_soap-25 | parsec_l | momentum | +0.1320 / +0.1416 / — / +0.1418 |
| searched_soap-50 | sgd | gradient | +0.0059 / +0.0125 / +0.0093 / +0.0125 |
| searched_soap-50 | sgd | momentum | -0.0034 / +0.0067 / — / +0.0067 |
| searched_soap-50 | soap_map | gradient | +0.0038 / +0.0146 / +0.0038 / +0.0147 |
| searched_soap-50 | soap_map | momentum | +0.0035 / +0.0149 / — / +0.0154 |
| searched_soap-50 | exact_ggn | gradient | -0.0081 / +0.0288 / +0.0290 / +0.0293 |
| searched_soap-50 | exact_ggn | momentum | +0.0456 / +0.0487 / — / +0.0491 |
| searched_soap-50 | ef_newton | gradient | -0.0164 / +0.0298 / — / +0.0299 |
| searched_soap-50 | ef_newton | momentum | +0.0240 / +0.0244 / — / +0.0247 |
| searched_soap-50 | parsec_l | gradient | -0.0349 / +0.0151 / +0.0151 / +0.0151 |
| searched_soap-50 | parsec_l | momentum | +0.0010 / +0.0185 / — / +0.0185 |
| exact_tensor_ggn-10 | sgd | gradient | +0.0080 / +0.0111 / +0.0089 / +0.0111 |
| exact_tensor_ggn-10 | sgd | momentum | +0.0060 / +0.0073 / — / +0.0076 |
| exact_tensor_ggn-10 | soap_map | gradient | +0.0145 / +0.0196 / +0.0145 / +0.0197 |
| exact_tensor_ggn-10 | soap_map | momentum | +0.0108 / +0.0128 / — / +0.0143 |
| exact_tensor_ggn-10 | exact_ggn | gradient | +0.0368 / +0.0582 / +0.0578 / +0.0583 |
| exact_tensor_ggn-10 | exact_ggn | momentum | +0.0334 / +0.0338 / — / +0.0349 |
| exact_tensor_ggn-10 | ef_newton | gradient | +0.0488 / +0.0611 / — / +0.0612 |
| exact_tensor_ggn-10 | ef_newton | momentum | +0.0172 / +0.0159 / — / +0.0172 |
| exact_tensor_ggn-10 | parsec_l | gradient | +0.0503 / +0.0684 / +0.0667 / +0.0687 |
| exact_tensor_ggn-10 | parsec_l | momentum | +0.0476 / +0.0514 / — / +0.0532 |
| exact_tensor_ggn-25 | sgd | gradient | +0.0123 / +0.0128 / +0.0116 / +0.0128 |
| exact_tensor_ggn-25 | sgd | momentum | +0.0000 / +0.0000 / — / +0.0000 |
| exact_tensor_ggn-25 | soap_map | gradient | +0.0167 / +0.0211 / +0.0170 / +0.0211 |
| exact_tensor_ggn-25 | soap_map | momentum | -0.0067 / +0.0000 / — / +0.0000 |
| exact_tensor_ggn-25 | exact_ggn | gradient | +0.0127 / +0.0399 / +0.0377 / +0.0402 |
| exact_tensor_ggn-25 | exact_ggn | momentum | -0.0156 / +0.0123 / — / +0.0144 |
| exact_tensor_ggn-25 | ef_newton | gradient | -0.0009 / +0.0405 / — / +0.0408 |
| exact_tensor_ggn-25 | ef_newton | momentum | -0.0116 / +0.0061 / — / +0.0073 |
| exact_tensor_ggn-25 | parsec_l | gradient | -0.0060 / +0.0236 / +0.0182 / +0.0236 |
| exact_tensor_ggn-25 | parsec_l | momentum | -0.0151 / +0.0013 / — / +0.0036 |
| exact_tensor_ggn-50 | sgd | gradient | +0.0154 / +0.0087 / +0.0139 / +0.0154 |
| exact_tensor_ggn-50 | sgd | momentum | +0.0051 / -0.0004 / — / +0.0052 |
| exact_tensor_ggn-50 | soap_map | gradient | +0.0196 / +0.0197 / +0.0197 / +0.0201 |
| exact_tensor_ggn-50 | soap_map | momentum | +0.0049 / +0.0056 / — / +0.0060 |
| exact_tensor_ggn-50 | exact_ggn | gradient | +0.0216 / +0.0391 / +0.0262 / +0.0400 |
| exact_tensor_ggn-50 | exact_ggn | momentum | -0.0309 / +0.0000 / — / +0.0009 |
| exact_tensor_ggn-50 | ef_newton | gradient | -0.0019 / +0.0364 / — / +0.0365 |
| exact_tensor_ggn-50 | ef_newton | momentum | -0.0138 / +0.0000 / — / +0.0005 |
| exact_tensor_ggn-50 | parsec_l | gradient | -0.0032 / +0.0151 / +0.0026 / +0.0163 |
| exact_tensor_ggn-50 | parsec_l | momentum | -0.0185 / +0.0000 / — / +0.0001 |

| State | RHS | Fixed step (own, held-out all, share): SOAP | PARSEC-L | PARSEC-L blend 1 |
| --- | --- | --- | --- | --- |
| searched_soap-10 | gradient | 0.0501, +0.0310, +0.62 | 0.0038, +0.0025, +0.66 | 0.0100, +0.0066, +0.66 |
| searched_soap-10 | momentum | 0.0499, +0.0320, +0.64 | 0.0065, +0.0055, +0.85 | 0.0082, +0.0070, +0.85 |
| searched_soap-25 | gradient | 0.0254, +0.0178, +0.70 | 0.0035, +0.0018, +0.51 | 0.0073, +0.0037, +0.51 |
| searched_soap-25 | momentum | 0.0157, +0.0098, +0.62 | 0.0046, +0.0037, +0.81 | 0.0054, +0.0044, +0.81 |
| searched_soap-50 | gradient | 0.0133, +0.0065, +0.49 | 0.0019, +0.0004, +0.22 | 0.0034, +0.0007, +0.22 |
| searched_soap-50 | momentum | 0.0087, +0.0041, +0.47 | 0.0024, +0.0013, +0.53 | 0.0028, +0.0015, +0.53 |
| exact_tensor_ggn-10 | gradient | 0.0175, -0.0396, -2.26 | 0.0136, +0.0071, +0.52 | 0.0267, +0.0138, +0.52 |
| exact_tensor_ggn-10 | momentum | 0.0200, -0.0116, -0.58 | 0.0183, +0.0151, +0.83 | 0.0212, +0.0174, +0.82 |
| exact_tensor_ggn-25 | gradient | 0.0399, +0.0202, +0.51 | 0.0110, +0.0037, +0.34 | 0.0192, +0.0063, +0.33 |
| exact_tensor_ggn-25 | momentum | 0.0013, -0.0112, -8.86 | 0.0059, +0.0015, +0.26 | 0.0066, +0.0016, +0.25 |
| exact_tensor_ggn-50 | gradient | 0.0331, +0.0207, +0.63 | 0.0064, +0.0013, +0.21 | 0.0116, +0.0024, +0.20 |
| exact_tensor_ggn-50 | momentum | 0.0128, +0.0064, +0.50 | 0.0043, -0.0005, -0.12 | 0.0047, -0.0006, -0.13 |
