

import sys
import json as json
import ast

def PrintUsage():
    sys.stderr.write("""
Usage:
    parse_python.py <file> [--chunk-size N] [--output-file OUTPUT]

Options:
    --chunk-size N    Process in chunks of N lines (default: 1000)
    --output-file F   Write output to file F instead of stdout

""")
    exit(1)

def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s

def parse_code_string(code_string, filename='<string>'):
    """Parse a Python code string and return JSON AST representation."""
    tree = ast.parse(code_string, filename)
    
    json_tree = []
    def gen_identifier(identifier, node_type = 'identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos
    
    def traverse_list(l, node_type = 'list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos
        
    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s.decode('utf-8')
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)
        

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr))
            if node.optional_vars:
                children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.TryExcept):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.TryFinally):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.finalbody, 'finalbody'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                children.append(traverse_list([node.name], 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child, ast.boolop) or isinstance(child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))
                
        if (len(children) != 0):
            json_node['children'] = children
        return pos
    
    traverse(tree)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)

def parse_file(filename):
    """Parse a Python file and return JSON AST representation."""
    return parse_code_string(read_file_to_string(filename), filename)

def process_file_chunked(input_file, output_file=None, chunk_size=1000):
    """
    Process a file in chunks. If the file appears to be JSONL (one JSON per line),
    process line by line. Otherwise, treat as Python source and parse normally.
    """
    output_handle = open(output_file, 'w') if output_file else sys.stdout
    processed = 0
    errors = 0
    
    try:
        with open(input_file, 'rt') as f:
            # Check if file is JSONL format (each line is a JSON object)
            first_line = f.readline()
            f.seek(0)  # Reset to beginning
            
            if first_line.strip().startswith('['):
                # JSONL format - process line by line
                sys.stderr.write("Processing as JSONL format (one JSON object per line)...\n")
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # Just pass through JSONL lines (they're already parsed)
                        output_handle.write(line + '\n')
                        processed += 1
                        if processed % chunk_size == 0:
                            sys.stderr.write("Processed %d lines...\n" % processed)
                    except Exception as e:
                        errors += 1
                        if errors <= 10:  # Only show first 10 errors
                            sys.stderr.write("Error on line %d: %s\n" % (line_num, str(e)))
            else:
                # Python source file - try to parse
                sys.stderr.write("Processing as Python source file...\n")
                try:
                    result = parse_file(input_file)
                    output_handle.write(result + '\n')
                    processed = 1
                except Exception as e:
                    sys.stderr.write("Error parsing file: %s\n" % str(e))
                    errors = 1
    finally:
        if output_file and output_handle != sys.stdout:
            output_handle.close()
    
    sys.stderr.write("Done. Processed: %d, Errors: %d\n" % (processed, errors))

if __name__ == "__main__":
    chunk_size = 1000
    output_file = None
    input_file = None
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--chunk-size' and i + 1 < len(sys.argv):
            chunk_size = int(sys.argv[i + 1])
            i += 2
        elif arg == '--output-file' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg.startswith('--'):
            sys.stderr.write("Unknown option: %s\n" % arg)
            PrintUsage()
        else:
            input_file = arg
            i += 1
    
    if not input_file:
        PrintUsage()
    
    # Process the file
    try:
        process_file_chunked(input_file, output_file, chunk_size)
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        sys.stderr.write("Unicode error: %s\n" % str(e))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user\n")
    except Exception as e:
        sys.stderr.write("Error: %s\n" % str(e))
        import traceback
        traceback.print_exc()
