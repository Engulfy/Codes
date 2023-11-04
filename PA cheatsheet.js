PA ANS
function helper(s, i) {
    const c = char_at(s, i);
    return is_undefined(c)
    ? null
    : pair(c, () => helper(s, i + 1));
}
    
function char_stream(s) {
    return helper(s, 0);
}

function solve(n, constraints) {
    return accumulate((constraint, ss) =>
                    filter(s => char_at(s, head(constraint))=== tail(constraint), ss),
                                filter(s => string_length(s) === n, pa_words),constraints);
}

function eval_poly(poly) {
    function p(x) {
        return accumulate(
                (t, sum) => head(t) * math_pow(x, tail(t)) + sum,
                0,
                poly);
    }
    return p;
}

function add_poly(poly1, poly2) {
    if (is_null(poly1)) {
        return poly2;
    } else if (is_null(poly2)) {
        return poly1;
    } else {
        const coeff1 = head(head(poly1));
        const coeff2 = head(head(poly2));
        const exp1 = tail(head(poly1));
        const exp2 = tail(head(poly2));
        if (exp1 === exp2) {
            return coeff1 + coeff2 === 0
            ? add_poly(tail(poly1), tail(poly2))
            : pair(pair(coeff1 + coeff2, exp1),
            add_poly(tail(poly1), tail(poly2)));
        } else if (exp1 < exp2) {
            return pair(head(poly1), add_poly(tail(poly1), poly2));
        } else {
            return pair(head(poly2), add_poly(poly1, tail(poly2)));
        }   
    }
}

function multiply_poly(poly1, poly2) {
    return accumulate((p, q) => add_poly(p, q),null,
                            map(t1 => map(t2 => pair(head(t1) * head(t2),tail(t1) + tail(t2)),poly2),
                                                                                            poly1));
}

function alt_column_matrix(R, C) {
    const M = [];
    for (let r = 0; r < R; r = r + 1) {
        M[r] = [];
    }
    let count = 1;
    for (let c = 0; c < C; c = c + 1) {
        if (c % 2 === 0) {
            for (let r = 0; r < R; r = r + 1) {
                M[r][c] = count;
                count = count + 1;
            }
        } else {
            for (let r = R - 1; r >= 0; r = r - 1) {
                M[r][c] = count;
                count = count + 1;
            }
        }
    }
    return M;
}


PA ANS

function make_k_list(k, d) {
    if (d === 0) {
        return 0;
    } else {
        let klist = null;
        for (let i = 0; i < k; i = i + 1) {
            klist = pair(make_k_list(k, d - 1), klist);
        }
    return klist;
    }
}


function make_postfix_exp(bae) {
    const pfe = [];
    let next = 0;
    function convert(sub_bae) {
        if (is_number(sub_bae)) {
            pfe[next] = sub_bae;
            next = next + 1;
        } else {
            convert(sub_bae[0]);
            convert(sub_bae[2]);
            pfe[next] = sub_bae[1];
            next = next + 1;
            }
    }
    convert(bae);
    return pfe;
}


function eval_postfix_exp(pfe) {
    let next = array_length(pfe) - 1;
    function evaluate() {
        const token = pfe[next];
        next = next - 1;
        if (is_number(token)) {
            return token;
        } else {
            const op = token;
            const right = evaluate();
            const left = evaluate();
            if (op === "+") {
                return left + right;
            } else if (op === "-") {
                return left - right;
            } else if (op === "*") {
                return left * right;
            } else {
                return left / right;
            }
        }
    }
    return evaluate();
}


function delta_encode(L) {
    function encode(xs, prev) {
        return is_null(xs)
        ? null
        : pair(head(xs) - prev, encode(tail(xs), head(xs)));
    }
    return encode(L, 0);
}

function delta_decode(D) {
    function decode(xs, prev) {
        return is_null(xs)
        ? null
        : pair(prev + head(xs), decode(tail(xs), prev + head(xs)));
    }
    return decode(D, 0);
}


function runlength_encode(L) {
    function encode(val, count, next) {
        return is_null(next)
        ? list(count === 1 ? val : pair(val, count))
        : head(next) === val
        ? encode(val, count + 1, tail(next))
        : pair(count === 1 ? val : pair(val, count),
        encode(head(next), 1, tail(next)));
    }
    return is_null(L)
    ? null
    : encode(head(L), 1, tail(L));
}


function overlap_area(aar1, aar2) {
// [a, b] and [c, d] are the input intervals.
    function overlap_length(a, b, c, d) {
        return math_max(0, math_min(b, d) - math_max(a, c));
        }
    const x_overlap = overlap_length(
                                get_x(aar1), get_x(aar1) + get_width(aar1),
                                    get_x(aar2), get_x(aar2) + get_width(aar2));
    const y_overlap = overlap_length(
                                get_y(aar1), get_y(aar1) + get_height(aar1),
                                    get_y(aar2), get_y(aar2) + get_height(aar2));
    return x_overlap * y_overlap;
}


streamS

function sum_primes(a, b) {
    function iter(count, accum) {
        if (count > b) {
            return accum;
        } else if (is_prime(count)) {
            return iter(count + 1, count + accum);
        } else {
            return iter(count + 1, accum);
        }
    }
    return iter(a, 0);
}

streamed version;
function sum_primes(a, b) {
    return accumulate(
    (x, y) => x + y,
    0,
    filter(is_prime, enum_list(a, b))
    );
}

fibgen using streams
function fibgen(a, b) {
    return pair(a, () => fibgen(b, a + b));
}
const fibs = fibgen(0, 1);

const fibs =pair(0,
                    () => pair(1,
                        () => add_streams(
                                stream_tail(fibs),
                                            fibs)));

more and more streams
function more(a, b) {
    return (a > b)
    ? more(1, 1 + b)
    : pair(a, () => more(a + 1, b));
}
const more_and_more = more(1, 1);


replace in stream
function replace(s, a, b) {
    return is_null(s)
    ? null
    : pair((head(s) === a) ? b : head(s),
    () => replace(stream_tail(s), a, b));
}

function list_to_inf_stream(xs) {
    function helper(ys) {
        return is_null(ys)
        ? helper(xs)
        : pair(head(ys), () => helper(tail(ys)));
    }
    return is_null(xs) ? null : helper(xs);
}
const s = list_to_inf_stream(list(1, 2, 3));

function memo_fun(fun) {
    let already_run = false;
    let result = undefined;
    function mfun() {
        if (!already_run) {
            result = fun();
            already_run = true;
            return result;
        } else {
            return result;
        }
    }
    return mfun;
}

primes stream
const primes =pair(2,
                    () => stream_filter(is_prime,
                                integers_from(3)));
                                
sieve using streams
function sieve(s) {
    return pair(head(s),
            () => sieve(stream_filter(
                    x => !is_divisible(x, head(s)),
                                    stream_tail(s))));
}
const primes = sieve(integers_from(2));   

sqrt funct
function average(a, b) {
    return (a + b) / 2;
}
function improve(guess, x) {
    return average(guess, x / guess);
}
function sqrt_stream(x) {
    const guesses = pair(1.0,
                () => stream_map(
                        guess => improve(guess, x),
                                            guesses));
    return guesses;
}

1-1/3+1/5+...
function pi_summands(n) {
    return pair(1 / n,
            () => stream_map(x => -x,
                    pi_summands(n + 2)));
}


function zip_streams(s1, s2) {

    return pair(head(s1), () => zip_streams(s2, stream_tail(s1)));

}

function zip_list_streams(xs){
    return pair(head(head(xs)), 
                ()=> zip_list_streams(append(tail(xs), 
                                             list(stream_tail(head(xs))))));
}

function partial_sums1(s) {

    // YOUR SOLUTION HERE
    const list = integers_from(1);
    function n_sum(n){
        return n === 0 
        ? 0 
        : stream_ref(s, n - 1) + n_sum(n - 1);
    }
    
    return stream_map(n_sum, list);

}

function partial_sum(s){
    return pair(head(s), () => add_streams(stream_tail(s), partial_sum(s)));
}


function rick_the_rabbit ( n ) {
    return n < 0
    ? 0
    : n === 0
    ? 1
    : rick_the_rabbit ( n - 1) // Rick hops
    +
    rick_the_rabbit ( n - 2) // Rick skips
    +
    rick_the_rabbit ( n - 3); // Rick jumps
}

coin change
function first_denomination ( kinds_of_coins ) {
    return kinds_of_coins === 1 ? 5 :
    kinds_of_coins === 2 ? 10 :
    kinds_of_coins === 3 ? 20 :
    kinds_of_coins === 4 ? 50 :
    kinds_of_coins === 5 ? 100 : 0;
}
function cc ( amount , kinds_of_coins ) {
    return amount === 0
    ? 1
    : amount < 0 || kinds_of_coins === 0
    ? 0
    : cc ( amount -
    first_denomination( kinds_of_coins ) ,
    kinds_of_coins )
    +
    cc ( amount , kinds_of_coins - 1);
}

simplified rational number
function gcd(a, b) {
    return b === 0 ? a : gcd(b, a % b);
}
function make_rat(n, d) {
    const g = gcd(n, d);
    return pair(n / g, d / g);
}


//LISTS
iterative length counter
function length_iter(xs) {
    function len(ys, counted_so_far) {
    return is_null(ys)
    ? counted_so_far
    : len(tail(ys), counted_so_far + 1);
    }
return len(xs, 0);
}

reverse list
function reverse2(lst) {
    return is_null(lst)
    ? null
    : append(reverse2(tail(lst)),
    list(head(lst)));
}

function reverse3(xs) {
    function rev(original, reversed) {
        return is_null(original)
        ? reversed
        : rev(tail(original),
        pair(head(original), reversed));
    }
    return rev(xs, null);
}

scale list
function scale_list(xs, k) {
    return is_null(xs)
    ? null
    : pair(k * head(xs),
    scale_list(tail(xs), k));
}

//TREES
counting in tree
function count_data_items(tree) {
    return is_null(tree)
    ? 0
    : ( is_list(head(tree))
    ? count_data_items(head(tree))
    : 1 )
    +
    count_data_items(tail(tree));
}

scaling tree
function scale_tree(tree, k) {
    return map(sub_tree =>
                    !is_list(sub_tree)
                    ? k * sub_tree
                    : scale_tree(sub_tree, k),
                    tree);
}

map tree
function map_tree(f, tree) {
    return map(sub_tree =>
                    !is_list(sub_tree)
                    ? f(sub_tree)
                    : map_tree(f, sub_tree),
                    tree);
}


// LISTS
smallest element in list
function smallest(xs) {
    return accumulate((x, y) => x < y ? x : y,
    head(xs), tail(xs));
}

better drop funct
function drop(xs, n) {
                    return n === 0
                    ? xs
                    : drop(tail(xs),n-1);
            }

destructive append
function d_append(xs, ys) {
    if (is_null(xs)) {
        return ys;
    } else {
        set_tail(xs, d_append(tail(xs), ys));
        return xs;
    }
}

destructive map
function d_map(fun, xs) {
    if (!is_null(xs)) {
        set_head(xs, fun(head(xs)));
        d_map(fun, tail(xs));
    } else { }
}

//ARRAYS

create array from 1 to n
function array_1_to_n(n) {
    const a = [];
    function iter(i) {
        if (i < n) {
            a[i] = i + 1;
            iter(i + 1);
            }
            }
            iter(0);
    return a;
}

mapping array
function map_array(f, arr) {
    const len = array_length(arr);
    function iter(i) {
        if (i < len) {
        arr[i] = f(arr[i]);
        iter(i + 1);
    }
    }
    iter(0);
}

reverse array
function swap(A, i, j) {
    let temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}
function reverse_array(A) {
    const len = array_length(A);
    const half_len = math_floor(len / 2);
    for (let i = 0; i < half_len; i = i + 1) {
        swap(A, i, len - 1 - i);
    }
}

create matrix of row x col
function zero_matrix(rows, cols) {
    const M = [];
    for (let r = 0; r < rows; r = r + 1) {
        M[r] = [];
        for (let c = 0; c < cols; c = c + 1) {
            M[r][c] = 0;
        }
    }
    return M;
}

matrix mulitplication of 3 by 3
function matrix_multiply_3x3(A, B) {
    const M = [];
    for (let r = 0; r < 3; r = r + 1) {
        M[r] = [];
        for (let c = 0; c < 3; c = c + 1) {
            M[r][c] = 0;
            for (let k = 0; k < 3; k = k + 1) {
                M[r][c] = M[r][c] + A[r][k] * B[k][c];
            }
        }
    }
    return M;
}

// SEARCHING
finding element(returns true or false)
function linear_search(A, v) {
    const len = array_length(A);
    let i = 0;
    while (i < len && A[i] !== v) {
        i = i + 1;
    }
    return (i < len);
}

binary search of list(returns true or false)
function binary_search(A, v) {
    function search(low, high) {
        if (low > high) {
            return false;
        } else {
            const mid = math_floor((low + high) / 2);
            return (v === A[mid]) ||
            (v < A[mid]
            ? search(low, mid - 1)
            : search(mid + 1, high));
        }
    }
    return search(0, array_length(A) - 1);
}

loop version binary search
function binary_search(A, v) {
    let low = 0;
    let high = array_length(A) - 1;
    while (low <= high) {
        const mid = math_floor((low + high) / 2 );
        if (v === A[mid]) {
            break;
        } else if (v < A[mid]) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return (low <= high);
}


selection sort for array
function selection_sort(A) {
    const len = array_length(A);
    for (let i = 0; i < len - 1; i = i + 1) {
        let min_pos = find_min_pos(A, i, len - 1);
        swap(A, i, min_pos);
    }
}
function find_min_pos(A, low, high) {
    let min_pos = low;
    for (let j = low + 1; j <= high; j = j + 1) {
        if (A[j] < A[min_pos]) {
            min_pos = j;
        }
    }
    return min_pos;
}

insertion sort for array
function insertion_sort(A) {
    const len = array_length(A);
    for (let i = 1; i < len; i = i + 1) {
        let j = i - 1;
        while (j >= 0 && A[j] > A[j + 1]) {
            swap(A, j, j + 1);
            j = j - 1;
        }
    }
}


function merge_sort(A) {
    merge_sort_helper(A, 0, array_length(A) - 1);
}
function merge_sort_helper(A, low, high) {
    if (low < high) {
    const mid = math_floor((low + high) / 2);
    merge_sort_helper(A, low, mid);
    merge_sort_helper(A, mid + 1, high);
    merge(A, low, mid, high);
    }
}

function merge(A, low, mid, high) {
    const B = []; // temporary array
    let left = low;
    let right = mid + 1;
    let Bidx = 0;
    while (left <= mid && right <= high) {
        if (A[left] <= A[right]) {
            B[Bidx] = A[left];
            left = left + 1;
        } else {
            B[Bidx] = A[right];
            right = right + 1;
        }
        Bidx = Bidx + 1;
    }
    while (left <= mid) {
        B[Bidx] = A[left];
        Bidx = Bidx + 1;
        left = left + 1;
    }
    while (right <= high) {
        B[Bidx] = A[right];
        Bidx = Bidx + 1;
        right = right + 1;
    }
    for (let k = 0; k < high - low + 1; k = k + 1) {
        A[low + k] = B[k];
    }
}

memoization
function memoize(f) {
    const mem = [];
    function mf(x) {
        if (mem[x] !== undefined) {
        return mem[x];
    } else {
        const result = f(x);
        mem[x] = result;
        return result;
        }
    }
    return mf;
}

n choose k
function choose(n, k) {
    return k > n
    ? 0
    : k === 0 || k === n
    ? 1
    : choose(n - 1, k) + choose(n - 1, k - 1);
}


const mem = [];
function read(n, k) {
    return mem[n] === undefined
    ? undefined
    : mem[n][k];
}
function write(n, k, value) {
    if (mem[n] === undefined) {
    mem[n] = [];
    }
    mem[n][k] = value;
}

memoize n choose k
function mchoose(n, k) {
    if (read(n, k) !== undefined) {
        return read(n, k);
    } else {
        const result = k > n ? 0
        : k === 0 || k === n ? 1
        : mchoose(n - 1, k) +
        mchoose(n - 1, k - 1);
        write(n, k, result);
        return result;
    }
}


partial sum of streams
function partial_sums(s) {

    // YOUR SOLUTION HERE
    function add_streams(s1, s2) {
        return is_null(s1)
        ? s2
        : is_null(s2)
        ? s1
        : pair(head(s1) + head(s2),
                    () => add_streams(stream_tail(s1),
                                             stream_tail(s2)));
        }
    return pair(head(s),
                () => add_streams(stream_tail(s),
                                  partial_sums(s)));
}

// PA cheatsheet

/**
stream functions (eg add streams and stuff)

memoization

arrays

permutations

binary search search for max and min values where condition is true

graph/tree traversal

Metacircular stuff maybe
**/

// -----------------------------------------------------------------------
// stream functions (eg add streams and stuff)
function scale_stream(c, stream) {
    return stream_map(x => c * x, stream);
}

function mul_streams(a,b) {
    return pair(head(a) * head(b),
                    () => mul_streams(stream_tail(a), stream_tail(b)));
}

function add_streams(s1, s2) {
    if (is_null(s1)) {
        return s2;
    } else if (is_null(s2)) {
        return s1;
    } else {
        return pair(head(s1) + head(s2),
                    () => add_streams(stream_tail(s1),
                                      stream_tail(s2)));
    }
}

function zip_stream(ss) {
    return pair(head(head(ss)), 
                () => zip_stream(append(tail(ss), list(head(ss)))));
}

const ones = pair(1, () => ones);

// -----------------------------------------------------------------------
// memoization
const mem = [];

function read(r, c) {
    return mem[r] === undefined
    ? undefined
    : mem[r][c];
}
function write(r, c, value) {
    if (mem[r] === undefined) {
        mem[r] = [];
    }
    mem[r][c] = value;
}


// -----------------------------------------------------------------------
// arrays
function swap(A, i, j) {
    let temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

function reverse_array(A) {
    const len = array_length(A);
    const half_len = math_floor(len / 2);
    let i = 0;
    while (i < half_len) {
        const j = len - 1 - i;
        swap(A, i, j);
        i = i + 1;
    }
}


// -----------------------------------------------------------------------
// permutation
function permutations(ys) {
    return is_null(ys)
        ? list(null)
        : accumulate(append, null,
            map(x => map(p => pair(x, p),
                         permutations(remove(x, ys))),
                ys));
}

// permutations(list(1,2,3,4));

// -----------------------------------------------------------------------
// bin search
// bin search on array in ascending order
function condition(x) {
    // searching for 0 for example
    const target = -5;
    return x === target
           ? 0
           : x < target
           ? 1
           : -1;
}

function binary_search(arr, f) {
    let low = 0;
    let high = array_length(arr) - 1;
    
    while (low <= high) {
        const mid = math_floor((low + high) / 2);
        const res = f(arr[mid]);
        if (res === 0) {
            // whether its found
            return true;
        } else if (res < 0) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return false;
}

// binary_search([-3,-2,-1,0,1,2,3], condition);

// binary search for min/max index where a condition is true

const arr = [false,false,false,false,true,true,true,false,false];
// function should return 4 for min and 6 for max;
// boolean satisfiability on sorted arrays

// -----------------------------------------------------------------------
// binary tree traversal psuedo-code
/**
function pre_order(tree) {
    if (is_null(tree)) {
        return null or undefined or anything;
    }
    else {
        do_somthing(value);
        pre_order(left subtree);
        pre_order(right subtree);
    }
}

function in_order(tree) {
    if (is_null(tree)) {
        return null or undefined or anything;
    }
    else {
        pre_order(left subtree);
        do_somthing(value);
        pre_order(right subtree);
    }
}

function in_order(tree) {
    if (is_null(tree)) {
        return null or undefined or anything;
    }
    else {
        pre_order(left subtree);
        pre_order(right subtree);
        do_somthing(value);
    }
}
**/

/**
function dfs(graph, target) {
    if (is_null(graph) || visited(graph)) {
        return null or undefined or anything;
    } else if (graph_val === target) {
        return true or update some variable;
    } else {
        mark_visited(node)
        for (all neighbours) {
            if (dfs(neighbour)) {
                return true;
            }
        }
    }
}
**/

/**
let layer = [start];
function bfs(graph, target, layer) {
    temp = [];
    for (all nodes in layer) {
        if (node === target) {
            return true or update something;
        }
        for (all neighbours of node) {
            temp[n] = neighbour;
        }
    }
    return bfs(graph, target, temp);
}
**/









// LISTS

    // APPEND LIST
    function append(xs, ys) {
        return is_null(xs)
            ? ys
            : pair(head(xs), append2(tail(xs), ys));
    }
    
    // REVERSE LIST
    function reverse(xs) {
        function rev(original, reversed) {
            return is_null(original)
                ? reversed
                : rev(tail(original),
                    pair(head(original), reversed));
        }
        return rev(xs, null);
    }

    
    // MAP LIST
    function map(f, xs) {
        return is_null(xs)
            ? null
            : pair(f(head(xs)), map(f, tail(xs)));
    }

    // FILTER LIST
    function filter(pred, xs) {
        return is_null(xs)
            ? null
            : pred(head(xs))
            ? pair(head(xs), filter(pred, tail(xs)))
            : filter(pred, tail(xs));
    }
    
    // ACCUMULATE LIST
    function accumulate(f, initial, xs) {
        return is_null(xs)
            ? initial
            : f(head(xs),
                accumulate(f, initial, tail(xs))
                );
    }
    
    // SORTING
    
        // INSERTION SORT LIST
        function insert(x, xs) {
            return is_null(xs)
                ? list(x)
                : x <= head(xs)
                ? pair(x, xs)
                : pair(head(xs), insert(x, tail(xs)));
        }
        
        function insertion_sort(xs) {
            return is_null(xs)
                ? xs
                : insert(head(xs), insertion_sort(tail(xs)));
        }

        // SELECTION SORT LIST
        // find smallest element of non-empty list xs
        function smallest(xs) {
            return accumulate((x, y) => x < y ? x : y,
                        head(xs), 
                        tail(xs));
        }
        
        function selection_sort(xs) {
            if (is_null(xs)) {
                return xs;
            } else {
                const x = smallest(xs);
                return pair(x, selection_sort(remove(x, xs)));
            }
        }

        // MERGESORT LIST
        
            function middle(n) {
                return math_floor(n / 2);
            }
    
            // put the first n elements of xs into a list
            function take(xs, n) {
                return n === 0
                    ? null
                    : pair(head(xs), take(tail(xs), n - 1));
            }
            
            // drop the first n elements from list, return rest
            function drop(xs, n) {
                    return n === 0
                    ? xs
                    : n === length(xs)
                    ? list()
                    : member(list_ref(xs, n), xs);
            }
            
            function merge(xs, ys) {
                if (is_null(xs)) {
                    return ys;
                } else if (is_null(ys)) {
                    return xs;
                } else {
                    const x = head(xs);
                    const y = head(ys);
                    return x < y
                           ? pair(x, merge(tail(xs), ys))
                           : pair(y, merge(xs, tail(ys)));
                }
            }
        
            function merge_sort(xs) {
                if (is_null(xs) || is_null(tail(xs))) {
                    return xs;
                } else {
                    const mid = middle(length(xs));
                    return merge(merge_sort(take(xs, mid)),
                                 merge_sort(drop(xs, mid)));
                }
            }
            
        // BUBBLE SORT ARRAY
        function bubblesort_array(A) {
            const len = array_length(A);
            for (let i = len - 1; i >= 1; i = i - 1) {
                for (let j = 0; j < i; j = j + 1) {
                    if (A[j] > A[j + 1]) {
                        const temp = A[j];
                        A[j] = A[j + 1];
                        A[j + 1] = temp;
                    }
                }
            }
        }
        
        // QUICK SORT LIST
        function partition(xs, p) {
            return pair(filter(x => x <= p, xs), filter(x => x > p, xs));
        }
        function quicksort(xs) {
            if (is_null(xs)) {
                return null;
            } else {
                const pair_of_lists = partition(tail(xs), head(xs));
                return append(quicksort(head(pair_of_lists)), 
                              pair(head(xs),
                                   quicksort(tail(pair_of_lists))));
            }
        }

//-------------------------------------------------------------------------

// TREES

    // COUNTING TREE
    function count_data_items(tree) {
        return is_null(tree)
            ? 0
            : ( is_list(head(tree))
                ? count_data_items(head(tree))
                : 1 )
                +
                count_data_items(tail(tree));
    }
    
    // MAP TREE
    function map_tree(f, tree) {
        return map(sub_tree =>
                        !is_list(sub_tree)
                        ? f(sub_tree)
                        : map_tree(f, sub_tree),
                    tree);
    }
    
    // ACCUMULATE TREE
    function accumulate_tree(f, op, initial, tree) {
        return accumulate((x, y) => is_list(x)
                       ? op(accumulate_tree(f, op, initial, x), y)
                       : op(f(x), y),
                      initial,
                      tree);
    }
    
    //E.G.
    function tree_sum(tree) {
        return accumulate_tree(x => x, (x, y) => x + y, 0 , tree);
    }

// ARRAYS / MUTABLE DATA

    // LINEAR SEARCH ARRAY
    function linear_search(A, v) {
        const len = array_length(A);
        let i = 0;
        while (i < len && A[i] !== v) {
            i = i + 1;
        }
        return (i < len);
    }

    // BINARY SEARCH ARRAY
    function binary_search(A, v) {
    
        function search(low, high) {
            if (low > high) {
                return false;
            } else {
                const mid = math_floor((low + high) / 2);
                return (v === A[mid]) ||
                       (v < A[mid] 
                        ? search(low, mid - 1)
                        : search(mid + 1, high));
            }
        }
        return search(0, array_length(A) - 1);
    }

    // READ AND WRITE
    const mem = [];
    
    function read(n, k) {
        return mem[n] === undefined
            ? undefined
            : mem[n][k];
    }
    
    function write(n, k, value) {
        
        if (mem[n] === undefined) {
            mem[n] = [];
        }
    
        mem[n][k] = value;
    }
    
    // MAP ARRAY
    function map_array(f, A) {
        const A_len = array_length(A);
        const arr = [];
    
        for (let i = 0; i < A_len; i = i + 1) {
            arr[i] = f(A[i]);
        }
    
        return arr;
    }

    // FILTER ARRAY
    function filter_array(pred, A) {
        const len = array_length(A);
        let res = [];
        let pos = 0;
        for (let i = 0; i < len; i = i + 1) {
            if (!pred(A[i])) {
                continue;
            } else {
                res[pos] = A[i];
                pos = pos + 1;
            }
        }
        return res;
    }

    // ACCUMUMULATE AREA (LEFT FOLDING)
    function accumulate_array(op, init, A) {
        const len = array_length(A);
        let acc = 0;
        for (let i = 0; i < len; i = i + 1) {
            if (i === 0) {
                acc = op(init, A[i]);
            } else {
                acc = op(acc, A[i]);
            }
        }
        return acc;
    }

    // SORTING
        // SELECTION SORT ARRAY
        function selection_sort(A) {
            const len = array_length(A);
    
            for (let i = 0; i < len - 1; i = i + 1) {
                let min_pos = find_min_pos(A, i, len - 1);
                swap(A, i, min_pos);
            }
        }
    
        function find_min_pos(A, low, high) {
            let min_pos = low;
            for (let j = low + 1; j <= high; j = j + 1) {
                if (A[j] < A[min_pos]) {
                    min_pos = j;
                }
            }
            return min_pos;
        }
    
        function swap(A, x, y) {
            const temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
        
        // INSERTION SORT ARRAY
        function insertion_sort(A) {
            function swap(A, x, y) {
                const temp = A[x];
                A[x] = A[y];
                A[y] = temp;
            }
            
            const len = array_length(A);
            
            for (let i = 1; i < len; i = i + 1) {
                let j = i - 1;
                
                while (j >= 0 && A[j] > A[j + 1]) {
                    swap(A, j, j + 1);
                    j = j - 1;
                }
            }
        }
        
        function insertion_sort2(A) {
            const len = array_length(A);
            
            for (let i = 1; i < len; i = i + 1) {
                const x = A[i];
                let j = i - 1;
                
                while (j >= 0 && A[j] > x) {
                    A[j + 1] = A[j]; // shift right
                    j = j - 1;
                }
                
                A[j + 1] = x;
            }
        }
        
        // MERGESORT ARRAY
        function merge_sort(A) {
            merge_sort_helper(A, 0, array_length(A) - 1);
        }
        
        function merge_sort_helper(A, low, high) {
            if (low < high) {
                const mid = math_floor((low + high) / 2);
                merge_sort_helper(A, low, mid);
                merge_sort_helper(A, mid + 1, high);
                merge(A, low, mid, high);
            }
        }
        
        function merge(A, low, mid, high) {
            const B = []; // temporary array
            let left = low;
            let right = mid + 1;
            let Bidx = 0;
            
            while (left <= mid && right <= high) {
                if (A[left] <= A[right]) {
                    B[Bidx] = A[left];
                    left = left + 1;
                } else {
                    B[Bidx] = A[right];
                    right = right + 1;
                }
                
            Bidx = Bidx + 1;
            }
            
            while (left <= mid) {
                B[Bidx] = A[left];
                Bidx = Bidx + 1;
                left = left + 1;
            }
            
            while (right <= high) {
                B[Bidx] = A[right];
                Bidx = Bidx + 1;
                right = right + 1;
            }
            
            for (let k = 0; k < high - low + 1; k = k + 1) {
                A[low + k] = B[k];
            }
        }
    
    // COPY ARRAY
    function copy_array(A) {
        const len = array_length(A);
        const B = [];
        for (let i = 0; i < len; i = i + 1) {
            B[i] = A[i];
        }
        return B;
    }
    
    // REVERSE ARRAY
    function reverse_array(A) {
        const len = array_length(A);
        const half_len = math_floor(len / 2);
        for (let i = 0; i < half_len; i = i + 1) {
            swap(A, i, len - 1 - i);
        }
    }
    
    // ARRAY TO LIST
    function array_to_list(A) {
        const len = array_length(A);
        let L = null;
        for (let i = len - 1; i >= 0; i = i - 1) {
            L = pair(A[i], L);
        }
        return L;
    }
    
    // LIST TO ARRAY
    function list_to_array(L) {
        const A = [];
        let i = 0;
        for (let p = L; !is_null(p); p = tail(p)) {
            A[i] = head(p);
            i = i + 1;
        }
        return A;
    }
    
    // EQUAL ARRAY
    function equal_array(A, B) {
        if (!is_array(A) || !is_array(B)) {
            return false;
        } else if (array_length(A) !== array_length(B)) {
            return false;
        } else {
            let is_equal = true;
            const len = array_length(A);
            for (let i = 0; is_equal && i < len; i = i + 1) {
                if (is_array(A[i]) || is_array(B[i])) {
                    is_equal = equal_array(A[i], B[i]);
                } else {
                    is_equal = equal(A[i], B[i]);
                }
            }
            return is_equal;
        }
    }
    
    // FLATTEN ARRAY
    function flatten(A) {
        if (is_number(A)) {
            return A;
        } else {
            const hd = map_array(flatten, A[0]);
            const other = [];
            
            for (let i = 1; i < array_length(A); i = i + 1) {
                other[i - 1] = A[i];
            }
            
            if (array_length(other) === 0) {
                return hd;
            } else {
                extend(hd, map_array(flatten, other));
                return hd;
            }
        }
    }
    
    flatten([[1, 2], [3, 4]]);
    
    // APPEND ARRAYS
    function append(A, item) {
        A[array_length(A)] = item;
    }

    // EXTEND ARRAY
    function extend(A, L) {
        if (array_length(L) === 0) {
            return A;
        }
        
        for (let i = 0; i < array_length(L); i = i + 1) {
            append(A, L[i]);
        }
    }
    
    // TRUNCATE ARRAY
    function truncate(A, items) {
        if (items < 0 || items > array_length(A)) {
            error("Cannot truncate beyond the length of the array");
        } else {
            const new_array = [];
            for (let i = 0; i < items; i = i + 1) {
                new_array[i] = A[i];
            }
            
            return new_array;
        }
    }
    
    // PAD AREA LEFT
    function pad_on_left(A, amount, padding) {
        if (amount < 0) {
            error("Cannot pad by a negative amount");
        } else {
            for (let i = amount + array_length(A) - 1; i >= 0; i = i - 1) {
                if (i > amount - 1) {
                    A[i] = A[i - amount];
                } else {
                    A[i] = padding;
                }
            }
        }
    }
    
    // PAD ARRAY RIGHT
    function pad_on_right(A, amount, padding) {
        if (amount < 0) {
            error("Cannot pad by a negative amount");
        } else {
            for (let i = 0; i < amount; i = i + 1) {
                append(A, padding);
            }
            
            return A;
        }
    }
    
    // ARRAY TO STRING
    function digits_to_string(digits) {
        const len = array_length(digits);
        let str = "";
        for (let i = 0; i < len; i = i + 1) {
            str = str + stringify(digits[i]);
        }
        return str;
    }
    
    // MEMOIZE
        function memoize(f) {
            const mem = [];
            
            function mf(x) {
                if (mem[x] !== undefined) {
                    return mem[x];
                } else {
                    const result = f(x);
                    mem[x] = result;
                    return result;
                }
            }
            
            return mf;
        }
        
        const mtrib =
            memoize(n => n === 0 ? 0
            : n === 1 ? 1
            : n === 2 ? 1
            : mtrib(n - 1) +
            mtrib(n - 2) +
            mtrib(n - 3));
            
        mtrib(23);

    // MATRIX
        // TRANSPOSE
        function transpose(M) {
            const rows = array_length(M);
            const cols = array_length(M[0]);
            let res = [];
            for (let r = 0; r < cols; r = r + 1) {
                res[r] = [];
                for (let c = 0; c < rows; c = c + 1) {
                    res[r][c] = M[c][r];
                }
            }
            return res;
        }
        
        // CREATE MATRIX CREATE 2D ARRAY i x j matrix
        function create_matrix(i, j, f) {
            const ans = [];
            const counter = 0;
        
            for (let x = 0; x < i; x = x + 1) {
                ans[x] = [];
                
                for (let y = 0; y < j; y = y + 1) {
                    ans[x][y] = f(counter);
                    counter = counter + 1;
                }
            }
            
            return ans;
        }

    //===============================================================
    function sprialMatrix(i, j) {
        // construct the empty matrix
        const ans = create_matrix(i, j, x => 0);
        
        // fill in the matrix
        function fillMatrix(ans, i, j, cur, dir){
            if (i < 0 || j < 0 || i >= array_length(ans) ||  j >= array_length(ans)) {
                return null;
            }
            
            if (ans[i][j] !== 0) {
                return null;
            }
            
            ans[i][j] = cur;
            
            // if moving upwards, fill in the table in the reverse direction
            if (dir === 'u') {
                fillMatrix(ans, i-1,j,cur+1, 'u');    
            }
            
            // explore all 4 directions
            fillMatrix(ans, i,j+1,cur+1, 'r');
            fillMatrix(ans, i+1,j,cur+1, 'd');
            fillMatrix(ans, i,j-1,cur+1, 'l');
            fillMatrix(ans, i-1,j,cur+1, 'u');
               
        }
        
        fillMatrix(ans, 0 , 0, 1,'r');
        
        return ans;
    }
    
// STREAMS
    // A stream with elements 1, 2, 3
    const s3 = pair(1,
                    () => pair(2,
                        () => pair(3,
                            () => null)));
    
    function ones_stream() {
            return pair(1, ones_stream);
    }
    
    // STREAM TAIL
    function stream_tail(stream) {
        return tail(stream)();
    }
    
    // ENUM STREAM
    function enum_stream(low, hi) {
        return low > hi
            ? null
            : pair(low,
                    () => enum_stream(low + 1, hi));
    }
    
    // STREAM REF
    function stream_ref(s, n) {
        return n === 0
            ? head(s)
            : stream_ref(stream_tail(s), n - 1);
    }
    
    // STREAM MAP
    function stream_map(f, s) {
        return is_null(s)
            ? null
            : pair(f(head(s)),
                    () => stream_map(f, stream_tail(s)));
    }
    
    
    // STREAM FILTER
    function stream_filter(p, s) {
        return is_null(s)
            ? null
            : p(head(s))
            ? pair(head(s),
                    () => stream_filter(p, stream_tail(s)))
            : stream_filter(p, stream_tail(s));
    }
    
    //EVAL STREAM
    function eval_stream(s, n) {
        return n === 0
            ? null
            : pair(head(s),
                eval_stream(stream_tail(s), n - 1));
    }
    
    // ADD STREAM
    function add_streams(s1, s2) {
        if (is_null(s1)) {
            return s2;
        } else if (is_null(s2)) {
            return s1;
        } else {
            return pair(head(s1) + head(s2),
                            () => add_streams(stream_tail(s1),
                                                stream_tail(s2)));
        }
    }
    
    // MULTIPLY STREAM
    function mul_streams(s1, s2) {
        if (is_null(s1)) {
            return s2;
        } else if (is_null(s2)) {
            return s1;
        } else {
            return pair(head(s1) * head(s2),
                        () => mul_streams(stream_tail(s1),
                                            stream_tail(s2)));
        }
    }
    
    // SCALE STREAMS
    function scale_stream(s, f) {
        return stream_map(x => x * f, s);
    }
    
    // PARTIAL SUM
    function partialsum(s) {
        return pair(head(s), 
                    () => add_stream(stream_tail(s), partialsum(s)));
    }