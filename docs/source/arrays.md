# Duck Array Support

Aggregating over other array types will work if the array types supports the following methods, [ufunc.reduceat](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html) or [ufunc.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)

| Reduction                      | `method="numpy"` | `method="flox"`   |
| ------------------------------ | ---------------- | ----------------- |
| sum, nansum                    | bincount         | add.reduceat      |
| mean, nanmean                  | bincount         | add.reduceat      |
| var, nanvar                    | bincount         | add.reduceat      |
| std, nanstd                    | bincount         | add.reduceat      |
| count                          | bincount         | add.reduceat      |
| prod                           | multiply.at      | multiply.reduceat |
| max, nanmax, argmax, nanargmax | maximum.at       | maximum.reduceat  |
| min, nanmin, argmin, nanargmin | minimum.at       | minimum.reduceat  |
