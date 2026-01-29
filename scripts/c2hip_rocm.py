#!/usr/bin/env python3
"""
C2HIP parser adapted from c2cuda.py for single-node single-GPU ROCm/HIP.
Based on FARGO3D c2cuda.py by Pablo Benitez Llambay, 2012-2014.

Key changes vs original:
- Generates HIP code: includes <hip/hip_runtime.h>
- Replaces cuda* runtime APIs with hip* equivalents
- Kernel launch uses hipLaunchKernelGGL instead of <<<grid,block>>>
- Cache config uses hipFuncSetCacheConfig + hipFuncCachePreferL1
- Profiling uses hipEvent_t APIs (optional)
- Output extension can be .cu / .hip / .cpp (hipcc can compile .cu too)
"""

from __future__ import print_function
import re
import sys
import getopt
import os

def verb(ifile, ofile):
    print('\nVERBOSE MODE ACTIVATED')
    print('=======================\n')
    print('\nInput file: ', ifile)
    print('Output file:',  ofile)

def read_file(input_file):
    try:
        ifile = open(input_file,'r')
    except IOError:
        print('\nI/O error in c2hip.py! Please, verify your input/output files.\n')
        sys.exit(1)
    return ifile.readlines()

def usage():
    print('\nUsage: -i --input=  --> input_file')
    print('       -o --output= --> output file (.cu/.hip/.cpp recommended)')
    print('       -v --verbose --> verbose mode')
    print('       -f --formatted --> formatted with astyle (external dependence)')
    print('       -p --profiling --> for block-dim studies (HIP events).\n')
    print('       -s --setup --> setup name.\n')
    sys.exit(1)

def opt_reader():
    # default values:
    verbose = False
    formated = False

    try:
        options, remainder = getopt.getopt(
            sys.argv[1:],
            'i:o:s:vfp',
            ['input=','output=','verbose','formated','profiling','setup=']
        )
    except getopt.GetoptError:
        usage()

    if options == []:
        usage()

    o_file = i_file = ''

    global profiling
    global SETUP
    global INPUT

    SETUP = ""
    profiling = False

    for opt, arg in options:
        if opt in ('-o', '--output'):
            o_file = arg
            continue
        if opt in ('-i', '--input'):
            i_file = arg
            INPUT = arg
            continue
        if opt in ('-v', '--verbose'):
            verbose = True
            continue
        if opt in ('-f', '--formated'):
            formated = True
            continue
        if opt in ('-p', '--profiling'):
            profiling = True
            continue
        if opt in ('-s', '--setup'):
            SETUP = arg
            continue

    opt = {
        'verbose': verbose,
        'input': i_file,
        'output': o_file,
        'formated': formated,
        'profiling': profiling,
        'setup': SETUP
    }
    return opt

def literal(lines, option, verbose=False):
    found = False
    output = []

    begin = '//<'   + option + '>'
    end   = '//<\\' + option + '>'

    if verbose:
        print('\n---------------------------------')
        print('Looking for ', option, ' lines.\n')

    for line in lines:
        line = line[:-1]  # Avoiding \n
        if line == begin:
            found = True
            continue

        if line == end:
            if verbose:
                if output == []:
                    print(option, ' is empty...')
                print('\nAll ' + option +  ' lines were stored.')
                print('---------------------------------\n')
            return output

        if found:
            output.append(line)
            if verbose:
                print(line[:-1], 'is a/an ' + option + ' line.')

def main_func(lines, verbose=False, test=False):
    if verbose:
        print('\n---------------------------------')
        print('Searching cpu main function...\n')

    function = re.compile(r"""
               (\w+)         #function type "1"
               \s+           #1 or more whitespace
               (\w+)_cpu     #function name "2"
               (\s?|\s+)     #1 or more whitespace
               \( (.*) \)    #input variables (all of them) "4"
               """, re.VERBOSE)

    if test:
        print('\nTEST OF MAIN_FUNC')
        print('=================\n')

        test_lines = [
            'void function_cpu (real dt, float b, string str_1) {',
            'void function_cpu (real dt, float b){',
            'void function_cpu(real dt, float b){',
            'void function_cpu(real dt, float b, int j) {',
            'void function_cpu (real dt,float b) {',
            'void function_cpu(real dt,float b){',
            'void     function_cpu    (real dt,   float b)    {'
        ]
        for line in test_lines:
            s = function.search(line)
            if s:
                func_type = s.group(1)
                func_name = s.group(2)
                func_var  = re.sub(',(\s+|\s?)',', ',s.group(4))
                parsed_line = func_type + ' ' + func_name
                parsed_line += '_gpu' + '(' + func_var + ') {\n'
                print(line, " was parsed as ")
                print(parsed_line)
        sys.exit(0)

    for line in lines:
        s = function.search(line)
        if s:
            func_type = s.group(1)
            func_name = s.group(2)
            func_var  = re.sub(',(\s+|\s?)',', ',s.group(4))
            parsed_line = func_type + ' ' + func_name
            parsed_line += '_gpu' + '(' + func_var + ') {\n'
            if verbose:
                print(line[:-1], " was parsed as ")
                print(parsed_line)
                print("Function", func_name, "was found.")
                print('---------------------------------')
            return parsed_line

def gathering_data(lines, verbose):
    flags      = literal(lines,'FLAGS',verbose)
    includes   = literal(lines,'INCLUDES',verbose)
    user_def   = literal(lines,'USER_DEFINED',verbose)
    loop       = literal(lines,'LOOP_VARIABLES',verbose)
    external   = literal(lines,'EXTERNAL',verbose)
    variables  = literal(lines,'VARIABLES',verbose)
    filling    = literal(lines,'FILLING_VARIABLES',verbose)
    internal   = literal(lines,'INTERNAL',verbose)
    main_loop  = literal(lines,'MAIN_LOOP',verbose)
    constant   = literal(lines,'CONSTANT',verbose)
    last_block = literal(lines,'LAST_BLOCK',verbose)
    gpu_func   = main_func(lines,test=False,verbose=verbose)

    data = {
        'flags': flags, 'includes': includes, 'user_def': user_def,
        'loop': loop,'external': external,'variables': variables,
        'filling': filling,'main_loop': main_loop,'gpu_func': gpu_func,
        'constant': constant, 'internal': internal, 'last_block': last_block
    }
    return data

def make_flags(flags):
    new_flags = []
    for element in flags:
        new_flags.append(element[2:])
    return new_flags

def parsing_external(external):
    if external is None:
        return [], [], []
    declarations = []
    calls = []
    variables = re.compile(r"""
                \s+                #1 or more whitespace
                (\w+\*?)           #variable type "1"
                \s+                #1 or more whitespace
                (\w+)              #variable name "2"
                (\s?|\s+)=         #1 or more whitespace and = "3"
                (\s?|\s+)          #1 or more whitespace and = "4"
                (.*);              #asign arguments "5"
                """, re.VERBOSE)

    externals = []

    for element in external:
        if not re.search("\s?//", element):  # Avoiding comments...
            if element and element[0] == '#':
                declarations.append(element)
                calls.append(element)
                continue
            s = variables.search(element)
            if not s:
                continue
            declarations.append(s.group(1)+' '+s.group(2))
            if re.match(".*_cpu", s.group(5)):
                calls.append(re.sub("_cpu","_gpu", s.group(5)))
            else:
                calls.append(s.group(5))
            externals.append([s.group(2), s.group(5)])

    return declarations, calls, externals

def _extract_launcher_varnames(gpu_func):
    """
    Extract argument *names* from gpu_func signature.
    Also return a set of arguments whose type includes 'Field'.
    """
    launcher = re.search(r"\w*\s*(.*)", gpu_func).group(1)  # avoiding return type
    variables = re.search(r"\((.*)\)", launcher).group(1).split(',')
    var = []
    field_args = set()
    for i in variables:
        parts = i.strip().split()
        if len(parts) >= 2:
            name = parts[-1].strip()
            var.append(name)
            if "Field" in i:
                field_args.add(name)
    return var, field_args

def make_launcher(gpu_func, calls):
    """
    Produce:
    - launcher body header: 'func_gpu(...) {'
    - a C-linkage signature line: 'extern "C" func_gpu(...) {'
    Launch is written later in make_output using hipLaunchKernelGGL.
    """
    launcher_sig = re.search(r"\w*\s*(.*)", gpu_func).group(1)  # avoiding type
    func_name = re.search(r"(.*)\(", launcher_sig).group(1)

    # We return launcher header and an extern "C" signature for the wrapper.
    return func_name, 'extern "C" ' + gpu_func

def kernel_name_from_gpu_func(gpu_func):
    """
    Derive the kernel function name from a gpu launcher signature.
    Example: "_ComputeForce_gpu(...)" -> "_ComputeForce_kernel"
    """
    launcher_sig = re.search(r"\w*\s*(.*)", gpu_func).group(1)  # drop return type
    func_name = re.search(r"(.*)\(", launcher_sig).group(1).strip()
    return re.sub(r"_gpu\b", "_kernel", func_name)

def make_kernel(gpu_func, declarations):
    launcher = re.sub("_gpu","_kernel",gpu_func)
    if re.search(r"\(\s*\)", launcher):
        launcher = "__global__ " + launcher[:-4] + '\n'
    else:
        launcher = "__global__ " + launcher[:-4] + ',\n'

    # Remove Field* args (FARGO3D convention)
    temporal1 = re.search(r"\(.*", launcher).group(0)
    temporal1 = re.sub(r"Field.*", " ", temporal1)
    launcher = re.search(r"(.*)\(", launcher).group(1) + temporal1

    for element in declarations:
        if element and element[0] == '#':
            launcher += element + '\n'
        else:
            launcher += element + ',\n'
    return launcher[:-2] + ") {\n"

def make_constant(symbols):
    data = re.compile(r"""
               (\s?|\s+)//       #comments "1"
               (\s?|\s+)(\w+)    #type "3"
               (\s?|\s+)(\w+)    #name "5"
               (\s?|\s+)\(?      #whitespace? "6"
               (.*)\);           #size "7"
               """, re.VERBOSE)

    cte_cpy = ''
    sizes = []

    for line in symbols:
        s = data.search(line)
        if not s:
            continue
        try:
            size = int(s.group(7))
        except ValueError:
            size = 796
        if size == 1:
            cte_cpy += 'hipMemcpyToSymbol(HIP_SYMBOL(' + \
                       s.group(5) + '_s), ' + \
                       "&" + s.group(5) + ', ' + \
                       'sizeof(' + s.group(3) + ')' + \
                       '*(' + s.group(7) + '), ' + \
                       '0, hipMemcpyHostToDevice);\n'
        else:
            # Match original CUDA path: use CUDAMEMCPY (maps to cudaMemcpyToSymbol)
            # Our cuda_compat.h maps cudaMemcpyToSymbol -> hipMemcpyToSymbol
            if s.group(5) == 'InvDiffXmed':
                # ROCm workaround: copy from host buffer to constant to avoid D2D symbol copy failure
                cte_cpy += 'hipMemcpyToSymbol(HIP_SYMBOL(' + \
                           s.group(5) + '_s), ' + \
                           s.group(5) + ', ' + \
                           'sizeof(' + s.group(3) + ')' + \
                           '*(' + s.group(7) + '), ' + \
                           '0, hipMemcpyHostToDevice);\n'
            else:
                cte_cpy += 'CUDAMEMCPY(' + \
                           s.group(5) + '_s, ' + \
                           s.group(5) + '_d, ' + \
                           'sizeof(' + s.group(3) + ')' + \
                           '*(' + s.group(7) + '), ' + \
                           '0, cudaMemcpyDeviceToDevice);\n'
        sizes.append(s.group(7))

    # determining size of constant memory....
    numvar = len(symbols)
    exact_size = 0
    vectors = 0
    for i in sizes:
        try:
            exact_size += int(i)
        except ValueError:
            vectors += 1
    try:
        vector_size = int((15384/2 - exact_size) / vectors)
    except ZeroDivisionError:
        vector_size = 0

    cte_dec = ''
    defines = ''
    undefs  = ''

    for line in symbols:
        s = data.search(line)
        if not s:
            continue
        try:
            size = int(s.group(7))
        except ValueError:
            size = vector_size
        if size > 1:
            cte_dec += 'CONSTANT(' + s.group(3)+ ', ' + s.group(5) + '_s, ' + str(size) + ');\n'
            defines += '#define ' + s.group(5) + "(i) " + s.group(5) + "_s[(i)]\n"
        else:
            cte_dec += '__device__ __constant__ ' + s.group(3) + ' ' + s.group(5) + '_s;\n'
            defines += '#define ' + s.group(5) + " " + s.group(5) + "_s\n"
            undefs  += '#undef ' + s.group(5) + '\n'

    return cte_cpy, cte_dec, defines, undefs

def make_mainloop(mainloop):
    data = re.compile(r"""
               (\s?|\s+)for\s*\(                 #identifying a for "1"
               (\s?|\s+)(.*)(\s+|\s?)=        #ivariable "3"
               (\s?|\s+)(.*)(\s+|\s?);        #lower index "6"
               (\s?|\s+).*(\s?|\s+)<
               (\s?|\s+)(.*)(\s?|\s+);        #upper index 11
               """, re.VERBOSE)
    var = []
    loop = False

    begin = '//<'   + '#' + '>'
    end   = '//<\\' + '#' + '>'

    effective_loop = []

    for line in mainloop:
        if data.search(line):
            s = data.search(line)
            var.append([s.group(3), s.group(6), s.group(11)])
        if line == begin:
            loop = True
            continue
        if line == end:
            loop = False
            break
        if loop:
            effective_loop.append(line)

    first_line = '#ifdef X \n' + \
        'i = threadIdx.x + blockIdx.x * blockDim.x;\n' + \
        '#else \n' + \
        'i = 0;\n' + \
        '#endif \n' + \
        '#ifdef Y \n' + \
        'j = threadIdx.y + blockIdx.y * blockDim.y;\n' + \
        '#else \n' + \
        'j = 0;\n' + \
        '#endif \n' + \
        '#ifdef Z \n' + \
        'k = threadIdx.z + blockIdx.z * blockDim.z;\n' + \
        '#else \n' + \
        'k = 0;\n' + \
        '#endif\n'

    second_line = ''
    second_line += '#ifdef Z\n'
    second_line += 'if(' + var[0][0] + '>=' + var[0][1] + \
        ' && ' + var[0][0] + '<' + var[0][2] + ') {\n'
    second_line += '#endif\n'
    second_line += '#ifdef Y\n'
    second_line += 'if(' + var[1][0] + '>=' + var[1][1] + \
        ' && ' + var[1][0] + '<' + var[1][2] + ') {\n'
    second_line += '#endif\n'
    second_line += '#ifdef X\n'
    second_line += 'if(' + var[2][0] + '<' + var[2][2] + ') {\n'
    second_line += '#endif\n'

    return first_line, second_line, effective_loop, var

def make_topology(var_loop, externals):
    blocks_define = ''

    BLOCKS = analyze_blocks()
    if BLOCKS is not None:
        blocks = 'dim3 block ({0:s}, {1:s}, {2:s});'.format(BLOCKS[0],BLOCKS[1],BLOCKS[2])
    else:
        blocks = 'dim3 block (BLOCK_X, BLOCK_Y, BLOCK_Z);'

    # original code assumes Nx/Ny/Nz macros in scope
    grid = 'dim3 grid ((' + "Nx+2*NGHX" + '+block.x-1)/block.x,\n' + \
           '((' + "Ny+2*NGHY" + ')+block.y-1)/block.y,\n' + \
           '((' + "Nz+2*NGHZ" + ')+block.z-1)/block.z);\n' + \
           'if (grid.x < 1) grid.x = 1;\n' + \
           'if (grid.y < 1) grid.y = 1;\n' + \
           '#ifndef Z\n' + \
           'grid.z = 1;\n' + \
           '#else\n' + \
           'if (grid.z < 1) grid.z = 1;\n' + \
           '#endif'

    return blocks_define, blocks, grid

def analyze_blocks():
    try:
        blocks = open("../setups/"+SETUP+"/"+SETUP+".blocks","r")
    except IOError:
        return None
    for line in blocks.readlines():
        split = line.split()
        search = re.search(split[0], INPUT[:-2])
        if search:
            BLOCK_X = split[1]
            BLOCK_Y = split[2]
            BLOCK_Z = split[3]
            return BLOCK_X, BLOCK_Y, BLOCK_Z
    return None

def make_output(f, output_file, formated=False):
    output = ''
    for line in f['flags']:
        output += line + '\n'

    output += '\n'

    # Provide safe defaults early (before headers that rely on them)
    output += r"""
#ifndef NFLUIDS
#define NFLUIDS 1
#endif
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif
#ifndef BLOCK_Z
#define BLOCK_Z 1
#endif
// Default coordinate system if none provided by flags/compile defs
#if !defined(CARTESIAN) && !defined(CYLINDRICAL) && !defined(SPHERICAL)
#define CYLINDRICAL
#endif
// Default dimensionality for standalone HIP compile (typical 2D fargo)
#if !defined(X) && !defined(Y) && !defined(Z)
#define X
#define Y
#endif
""" + "\n"

    # Ensure HIP runtime include exists (prepend if user didn't provide it)
    includes_text = "\n".join(f['includes']) + "\n"
    if "hip/hip_runtime.h" not in includes_text:
        output += '#include <hip/hip_runtime.h>\n'
    for line in f['includes']:
        output += line + '\n'

    # Minimal helper macros (guarded to avoid clashing with project headers)
    output += r"""
#ifndef HIPMEMCPY
  #define HIPMEMCPY(dst, src, sz, ofs, kind) hipMemcpy((dst), (src), (sz), (kind))
#endif

#ifndef HIP_CHECK_ERRORS_DEFINED
#define HIP_CHECK_ERRORS_DEFINED
static inline void hip_check_errors(const char* msg) {
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    fprintf(stderr, "HIP error after %s: %s\n", msg, hipGetErrorString(e));
    abort();
  }
}
#endif
""" + "\n"

    output += '\n' + f['defines']
    output += '\n' + f['cte_dec']
    output += '\n' + f['kernel'] + '\n'

    for line in f['internal']:
        output += line + '\n'

    output += '\n' + f['first_line']
    output += '\n' + f['second_line']

    for line in f['effective_loop']:
        output += line + '\n'

    output += '#ifdef X \n } \n #endif\n'
    output += '#ifdef Y \n } \n #endif\n'
    output += '#ifdef Z \n } \n #endif\n'

    output += '}\n'

    # launcher wrapper signature
    output += '\n' + f['def_launcher']

    output +=  '\n' + f['undefs'] + '\n'

    try:
        for line in f['user_def']:
            output += line + '\n'
    except TypeError:
        pass

    output += '\n' + f['blocks']
    output += '\n' + f['grid'] + '\n'

    # BIGMEM block is left as-is (project-specific macros/vars)
    output += '\n#ifdef BIGMEM\n'
    output += ('#define xmin_d &Xmin_d\n' + \
                   '#define ymin_d &Ymin_d\n' + \
                   '#define zmin_d &Zmin_d\n')
    output += ('#define Sxj_d &Sxj_d\n'+ \
                   '#define Syj_d &Syj_d\n'+ \
                   '#define Szj_d &Szj_d\n'+ \
                   '#define Sxk_d &Sxk_d\n'+ \
                   '#define Syk_d &Syk_d\n'+ \
                   '#define Szk_d &Szk_d\n'+ \
                   '#define Sxi_d &Sxi_d\n'+ \
                   '#define InvVj_d &InvVj_d\n'+ \
                   '#define InvDiffXmed_d &InvDiffXmed_d\n')
    output += '#endif\n'

    output += '\n' + f['cte_cpy'] + '\n'
    output += '\n#ifdef HIP_CHECK_AFTER_CTE\nhip_check_errors(\"cte_cpy\");\n#endif\n'

    # Cache config (HIP)
    kernel_name = f['kernel_name_kernel']
    cache = 'hipFuncSetCacheConfig((const void*)' + kernel_name + ', hipFuncCachePreferL1 );'
    output += '\n#ifndef HIP_DISABLE_CACHECONFIG\n' + cache + '\n#endif\n'

    # Build HIP launch statement
    # Map externals (lhs var name -> rhs expression, upgraded to *_gpu if needed)
    ext_map = {}
    for var, rhs in f.get('externals', []):
        if not var or not rhs:
            continue
        rhs_gpu = re.sub(r"_cpu\b", "_gpu", rhs)
        if var == "size_z":
            rhs_gpu = '((((Nz+2*NGHZ-2)>0 ? (Nz+2*NGHZ-2) : 1))>0 ? (((Nz+2*NGHZ-2)>0 ? (Nz+2*NGHZ-2) : 1)) : 1)'
        ext_map[var] = rhs_gpu

    def _kernel_param_chunks(kernel_code):
        lines = kernel_code.splitlines()
        start = None
        params = []
        for i, line in enumerate(lines):
            if "__global__" in line and "(" in line:
                start = i
                params.append(line.split("(", 1)[1])
                if ")" in line and not line.lstrip().startswith('#'):
                    params[-1] = params[-1].split(")", 1)[0]
                    return [c for c in "\n".join(params).split(',') if c.strip()]
                break
        if start is None:
            return []
        for line in lines[start+1:]:
            if ")" in line and not line.lstrip().startswith('#'):
                params.append(line.split(")", 1)[0])
                break
            params.append(line)
        return [c for c in "\n".join(params).split(',') if c.strip()]

    kernel_sig = f.get('kernel', '')
    kernel_chunks = _kernel_param_chunks(kernel_sig)

    extra_map = {
        'pitch': 'Pitch_gpu',
        'stride': 'Stride_gpu',
        'size_x': 'XIP',
        'size_y': 'Ny+2*NGHY-1',
        'size_z': '((((Nz+2*NGHZ-2)>0 ? (Nz+2*NGHZ-2) : 1))>0 ? (((Nz+2*NGHZ-2)>0 ? (Nz+2*NGHZ-2) : 1)) : 1)',
        'nx': 'Nx',
        'ny': 'Ny',
        'nz': 'Nz',
        'nghy': 'NGHY',
        'nghz': 'NGHZ',
    }

    argnames = []
    field_argnames = f.get('field_argnames', set())
    arg_lines = []
    if kernel_chunks:
        for chunk in kernel_chunks:
            for line in [l for l in chunk.splitlines() if l.strip()]:
                if line.lstrip().startswith('#'):
                    arg_lines.append(line.strip())
                    continue
                line = re.sub(r"__restrict__", "", line)
                tokens = line.replace('*', ' * ').split()
                if not tokens:
                    continue
                pname = tokens[-1].strip()
                if pname in ext_map:
                    expr = ext_map[pname]
                elif pname in field_argnames:
                    expr = pname + "->field_gpu"
                elif pname in extra_map:
                    expr = extra_map[pname]
                else:
                    expr = pname
                arg_lines.append(", " + expr)
    else:
        used_names = set()
        for name in f.get('gpu_argnames', []):
            name = name.strip()
            if not name:
                continue
            used_names.add(name)
            if name in field_argnames and name not in ext_map:
                argnames.append(name + "->field_gpu")
            else:
                argnames.append(ext_map.get(name, name))

        ext_iter = iter(f.get('externals', []))
        for element in f.get('calls', []):
            if element and element[0] == '#':
                continue
            try:
                var, rhs = next(ext_iter)
            except StopIteration:
                break
            if var in used_names:
                continue
            argnames.append(element)

        if not argnames:
            for element in f.get('calls', []):
                if element and element[0] == '#':
                    continue
                argnames.append(element)

    # remove empty / trailing commas safety
    argnames = [a.strip() for a in argnames if a and a.strip()]

    # HIP launch: hipLaunchKernelGGL(func, grid, block, sharedMem, stream, args...)
    launch = "hipLaunchKernelGGL({kname}, grid, block, 0, 0".format(kname=kernel_name)
    if arg_lines:
        for l in arg_lines:
            launch += "\n" + l
    else:
        for a in argnames:
            launch += ", " + a
    launch += ");\n"

    # Optional debug for addviscosity_cyl launch
    if kernel_name == "addviscosity_cyl_kernel":
        output += '\n#ifdef DEBUG_LAUNCH\n'
        output += 'fprintf(stderr, "addviscosity_cyl launch block=(%d,%d,%d) grid=(%d,%d,%d) size=(%d,%d,%d) Ny=%d NGHY=%d Nz=%d NGHZ=%d XIP=%d\\n", '\
                  'block.x, block.y, block.z, grid.x, grid.y, grid.z, '\
                  'XIP, Ny+2*NGHY-2, ((Nz+2*NGHZ-2)>0 ? (Nz+2*NGHZ-2) : 1), Ny, NGHY, Nz, NGHZ, XIP);\n'
        output += '#endif\n'

    output += '\n' + launch

    if not profiling:
        output += '\n' + 'hip_check_errors("' + kernel_name + '");\n'

    # Optional profiling block (HIP events)
    if profiling:
        prof = r"""
hipEvent_t start, stop;
float time;

hipEventCreate(&start);
hipEventCreate(&stop);

int eex, ey, ez;

#ifndef X
{eex=0;
#else
for (eex=3; eex < 7; eex++) {
#endif
#ifndef Y
{ey=0;
#else
for (ey=0; ey < 7; ey++) {
#endif
#ifndef Z
{ez=0;
#else
for (ez=0; ez < 7; ez++) {
#endif
  block.x = 1<<eex;
  block.y = 1<<ey;
  block.z = 1<<ez;
  if (block.x * block.y * block.z <= 1024) {
 grid.x  = (Nx+2*NGHX+block.x-1)/block.x;
 grid.y  = ((Ny+2*NGHY)+block.y-1)/block.y;
 grid.z  = ((Nz+2*NGHZ)+block.z-1)/block.z;

 hipEventRecord(start, 0);
"""
        output += '\n' + prof
        output += '\n' + launch
        prof2 = r"""
hipDeviceSynchronize();
hipError_t hipError = hipGetLastError();
hipEventRecord(stop, 0);
hipEventSynchronize(stop);
if (hipError == hipSuccess) {
  hipEventElapsedTime(&time, start, stop);
  printf ("%s\t%d\t%d\t%d\t%f\n", """ + '"' + output_file + '"' + r""", block.x, block.y, block.z, time);
}
}}}} 
exit(1);
"""
        output += prof2

    try:
        for line in f['last_block']:
            output += line +'\n'
    except TypeError:
        pass

    output += '\n}'

    if len(output_file) == 0:
        print(output)
    else:
        out = open(output_file,'w')
        out.write(output)

    if formated is True:
        os.system('astyle ' + output_file + "&")
    return

def main():
    options = opt_reader()

    verbose     = options['verbose']
    input_file  = options['input']
    output_file = options['output']

    if input_file == output_file:
        print("\nWARNING!!! You would overwrite your input file!!!")
        print("            This is not allowed...\n")
        sys.exit(1)

    if not (output_file.endswith('.cu') or output_file.endswith('.hip') or output_file.endswith('.cpp') or output_file.endswith('.cc') or output_file.endswith('.cxx')):
        print('\nWARNING!!! Your output file should be .cu/.hip/.cpp (hipcc can compile .cu)!!!\n')
        sys.exit(1)

    if verbose:
        verb(input_file, output_file)

    input_lines = read_file(input_file)

    data = gathering_data(input_lines, verbose)

    declarations, calls, externals  = parsing_external(data['external'])
    flags                           = make_flags(data['flags'])

    kernel_name, def_launcher       = make_launcher(data['gpu_func'], calls)
    kernel_name_kernel              = kernel_name_from_gpu_func(data['gpu_func'])
    kernel                          = make_kernel(data['gpu_func'], declarations)

    if data['constant'] is not None:
        cte_cpy, cte_dec, defines, undefs = make_constant(data['constant'])
    else:
        cte_cpy = cte_dec = defines = undefs = ''

    first_line, second_line, effective_loop, var_loop = make_mainloop(data['main_loop'])
    blocks_define, blocks, grid = make_topology(var_loop, externals)

    gpu_argnames, field_argnames = _extract_launcher_varnames(data['gpu_func'])

    final = {
        'flags': flags,
        'kernel': kernel,
        'includes': data['includes'],
        'defines': defines,
        'internal': data['internal'],
        'user_def': data['user_def'],
        'cte_cpy': cte_cpy,
        'cte_dec': cte_dec,
        'undefs': undefs,
        'first_line': first_line,
        'second_line': second_line,
        'effective_loop': effective_loop,
        'blocks_define': blocks_define,
        'blocks': blocks,
        'grid': grid,
        'last_block': data['last_block'],
        'def_launcher': def_launcher,
        'kernel_name': kernel_name,
        'kernel_name_kernel': kernel_name_kernel,
        'gpu_argnames': gpu_argnames,
        'field_argnames': field_argnames,
        'calls': calls,
        'externals': externals
    }

    make_output(final, output_file, formated=options['formated'])

if __name__=='__main__':
    main()
