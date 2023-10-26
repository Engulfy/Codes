function koo(fun, A) {
const len = array_length(A);
let m = 0;
while (m < len) {
A[m] = fun(A[m], A[m]);
m = m + 1;
}
}
function see(a, n) {
let acc = 0;
let i = 0;
while (i < n) {
let sum = acc + a;
acc = sum;
i = i + 1;
}
return acc;
}
const AA = [1, 2, 3, 4];
koo(see, AA);
AA;