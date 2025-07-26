package javabridge;

// javabridge/JavaParserBridge.java
import com.github.javaparser.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;

import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class JavaParserBridge {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: JavaParserBridge <command> <file> [method1,method2,...]");
            System.exit(1);
        }
        String command = args[0];
        String filePath = args[1];
        String code = new String(Files.readAllBytes(Paths.get(filePath)));
        JavaParser parser = new JavaParser();
        ParseResult<CompilationUnit> result = parser.parse(code);
        if (!result.isSuccessful() || !result.getResult().isPresent()) {
            System.err.println("Failed to parse file: " + filePath);
            System.exit(2);
        }
        CompilationUnit cu = result.getResult().get();
        // Remove all comments from the AST
        cu.getAllContainedComments().forEach(com.github.javaparser.ast.comments.Comment::remove);

        switch (command) {
            case "extract_methods":
                Set<String> methods = new HashSet<>();
                cu.findAll(MethodDeclaration.class).forEach(m -> {
                    if (m.isPublic()) methods.add(m.getNameAsString());
                });
                System.out.println(String.join(",", methods));
                break;
            case "extract_signatures": {
                String targetName = Paths.get(filePath).getFileName().toString().replace(".java", "");
                boolean found = false;
                for (TypeDeclaration<?> type : cu.getTypes()) {
                    if (type.getNameAsString().equals(targetName)) {
                        printTypeSignature(type);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    System.err.println("[WARN] No top-level type matching " + targetName + " in " + filePath + ". Outputting all top-level types.");
                    for (TypeDeclaration<?> type : cu.getTypes()) {
                        printTypeSignature(type);
                    }
                }
                break;
            }
            case "extract_minimal_class":
                if (args.length < 3) {
                    System.err.println("Usage: JavaParserBridge extract_minimal_class <file> method1,method2,...");
                    System.exit(1);
                }
                Set<String> targetMethods = new HashSet<>(Arrays.asList(args[2].split(",")));
                Set<String> usedFields = new HashSet<>();
                Set<String> usedTypes = new HashSet<>();
                Optional<ClassOrInterfaceDeclaration> mainClassOpt = cu.findFirst(ClassOrInterfaceDeclaration.class);
                if (!mainClassOpt.isPresent()) {
                    System.out.println("");
                    break;
                }
                ClassOrInterfaceDeclaration mainClass = mainClassOpt.get();
                Map<String, MethodDeclaration> methodMap = new HashMap<>();
                for (MethodDeclaration md : mainClass.getMethods()) {
                    methodMap.put(md.getNameAsString(), md);
                }
                Set<String> toVisit = new HashSet<>(targetMethods);
                Set<String> visited = new HashSet<>();
                while (!toVisit.isEmpty()) {
                    String m = toVisit.iterator().next();
                    toVisit.remove(m);
                    if (!visited.add(m)) continue;
                    MethodDeclaration md = methodMap.get(m);
                    if (md == null) continue;
                    md.accept(new VoidVisitorAdapter<Void>() {
                        @Override
                        public void visit(FieldAccessExpr n, Void arg) {
                            usedFields.add(n.getNameAsString());
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(MethodCallExpr n, Void arg) {
                            n.getScope().ifPresent(scope -> {
                                if (scope.isThisExpr() || scope.isSuperExpr()) {
                                    String called = n.getNameAsString();
                                    if (methodMap.containsKey(called) && !visited.contains(called)) {
                                        toVisit.add(called);
                                    }
                                } else if (scope.isNameExpr()) {
                                    // This captures field references like: benCompleteMapper.benDetailForOutboundDTOToIBeneficiary()
                                    String fieldName = scope.asNameExpr().getNameAsString();
                                    usedFields.add(fieldName);
                                }
                            });
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(ClassOrInterfaceType n, Void arg) {
                            usedTypes.add(n.getNameAsString());
                            super.visit(n, arg);
                        }
                    }, null);
                }
                cu.findAll(ClassOrInterfaceDeclaration.class).forEach(cls -> {
                    System.out.println(getClassSignature(cls) + " {");
                    for (FieldDeclaration field : cls.getFields()) {
                        boolean isUsed = false;
                        for (VariableDeclarator vd : field.getVariables()) {
                            if (usedFields.contains(vd.getNameAsString())) {
                                isUsed = true;
                                break;
                            }
                        }
                        if (isUsed) {
                            System.out.println("    " + field.toString().replace("\n", " ").trim());
                        }
                    }
                    for (MethodDeclaration method : cls.getMethods()) {
                        if (targetMethods.contains(method.getNameAsString()))
                            System.out.println("    " + method.toString().replace("\n", "\n    ").trim());
                    }
                    System.out.println("}");
                });
                break;
            case "extract_referenced_types":
                if (args.length < 3) {
                    System.err.println("Usage: JavaParserBridge extract_referenced_types <file> method1,method2,...");
                    System.exit(1);
                }
                Set<String> batchMethods = new HashSet<>(Arrays.asList(args[2].split(",")));
                Set<String> referencedTypes = new HashSet<>();
                Optional<ClassOrInterfaceDeclaration> mainClassOpt2 = cu.findFirst(ClassOrInterfaceDeclaration.class);
                if (!mainClassOpt2.isPresent()) {
                    System.out.println("");
                    break;
                }
                ClassOrInterfaceDeclaration mainClass2 = mainClassOpt2.get();
                Map<String, MethodDeclaration> methodMap2 = new HashMap<>();
                for (MethodDeclaration md : mainClass2.getMethods()) {
                    methodMap2.put(md.getNameAsString(), md);
                }
                // Recursively collect all methods called by the batch
                Set<String> toVisit2 = new HashSet<>(batchMethods);
                Set<String> visited2 = new HashSet<>();
                while (!toVisit2.isEmpty()) {
                    String m = toVisit2.iterator().next();
                    toVisit2.remove(m);
                    if (!visited2.add(m)) continue;
                    MethodDeclaration md = methodMap2.get(m);
                    if (md == null) continue;
                    // Collect types from signature
                    if (md.getType() != null) referencedTypes.add(md.getType().toString());
                    for (Parameter p : md.getParameters()) referencedTypes.add(p.getType().toString());
                    for (Type t : md.getThrownExceptions()) referencedTypes.add(t.toString());
                    // Collect types from body
                    md.accept(new VoidVisitorAdapter<Void>() {
                        @Override
                        public void visit(ObjectCreationExpr n, Void arg) {
                            referencedTypes.add(n.getType().toString());
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(VariableDeclarationExpr n, Void arg) {
                            referencedTypes.add(n.getElementType().toString());
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(FieldAccessExpr n, Void arg) {
                            referencedTypes.add(n.getScope().toString());
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(MethodCallExpr n, Void arg) {
                            n.getScope().ifPresent(scope -> {
                                if (scope.isThisExpr() || scope.isSuperExpr() || scope.isNameExpr()) {
                                    String called = n.getNameAsString();
                                    if (methodMap2.containsKey(called) && !visited2.contains(called)) {
                                        toVisit2.add(called);
                                    }
                                }
                            });
                            super.visit(n, arg);
                        }
                        @Override
                        public void visit(ClassOrInterfaceType n, Void arg) {
                            referencedTypes.add(n.getNameAsString());
                            super.visit(n, arg);
                        }
                    }, null);
                }
                // Also collect field types used by these methods
                for (FieldDeclaration fd : mainClass2.getFields()) {
                    for (VariableDeclarator vd : fd.getVariables()) {
                        referencedTypes.add(vd.getType().toString());
                    }
                }
                // Remove primitives and java.lang
                Set<String> filtered = referencedTypes.stream()
                    .filter(t -> !t.matches("int|long|double|float|boolean|char|byte|short|void|String"))
                    .collect(Collectors.toSet());
                System.out.println(String.join(",", filtered));
                break;
            default:
                System.err.println("Unknown command: " + command);
                System.exit(1);
        }
    }

    private static void printTypeSignature(TypeDeclaration<?> type) {
        if (type instanceof ClassOrInterfaceDeclaration) {
            ClassOrInterfaceDeclaration cls = (ClassOrInterfaceDeclaration) type;
            if (cls.isInterface()) {
                // Handle interface signature
                StringBuilder sb = new StringBuilder();
                if (cls.isPublic()) sb.append("public ");
                sb.append("interface ").append(cls.getNameAsString());
                if (cls.getExtendedTypes().size() > 0) {
                    sb.append(" extends ");
                    sb.append(cls.getExtendedTypes().stream().map(Object::toString).collect(Collectors.joining(", ")));
                }
                sb.append(" {");
                System.out.println(sb.toString());
                // Print method signatures
                for (MethodDeclaration method : cls.getMethods()) {
                    StringBuilder msig = new StringBuilder();
                    if (method.isPublic()) msig.append("    public ");
                    msig.append(method.getType().toString()).append(" ");
                    msig.append(method.getNameAsString()).append("(");
                    msig.append(method.getParameters().stream().map(Object::toString).collect(Collectors.joining(", ")));
                    msig.append(");");
                    System.out.println(msig.toString());
                }
                System.out.println("}");
            } else {
                // Existing class logic
                System.out.println(getClassSignature(cls) + " {");
                // Fields
                for (FieldDeclaration field : cls.getFields()) {
                    if (field.isPublic() || field.isProtected())
                        System.out.println("    " + field.toString().replace("\n", " ").trim());
                }
                // Constructors
                for (ConstructorDeclaration ctor : cls.getConstructors()) {
                    if (ctor.isPublic() || ctor.isProtected())
                        System.out.println("    " + getConstructorSignature(ctor) + ";");
                }
                // Methods
                for (MethodDeclaration method : cls.getMethods()) {
                    if (method.isPublic() || method.isProtected())
                        System.out.println("    " + getMethodSignature(method) + ";");
                }
                System.out.println("}");
            }
        } else if (type instanceof EnumDeclaration) {
            System.out.println("public enum " + type.getNameAsString() + " { ... }");
        } else if (type instanceof AnnotationDeclaration) {
            System.out.println("public @interface " + type.getNameAsString() + " { ... }");
        }
    }

    private static String getClassSignature(ClassOrInterfaceDeclaration cls) {
        StringBuilder sb = new StringBuilder();
        if (cls.isPublic()) sb.append("public ");
        if (cls.isInterface()) sb.append("interface ");
        else sb.append("class ");
        sb.append(cls.getNameAsString());
        if (cls.getExtendedTypes().size() > 0) {
            sb.append(" extends ");
            sb.append(cls.getExtendedTypes().stream().map(Object::toString).collect(Collectors.joining(", ")));
        }
        if (cls.getImplementedTypes().size() > 0) {
            sb.append(" implements ");
            sb.append(cls.getImplementedTypes().stream().map(Object::toString).collect(Collectors.joining(", ")));
        }
        return sb.toString();
    }

    private static String getMethodSignature(MethodDeclaration method) {
        StringBuilder sb = new StringBuilder();
        if (method.isPublic()) sb.append("public ");
        if (method.isProtected()) sb.append("protected ");
        if (method.isStatic()) sb.append("static ");
        sb.append(method.getType().toString()).append(" ");
        sb.append(method.getNameAsString()).append("(");
        sb.append(method.getParameters().stream().map(Object::toString).collect(Collectors.joining(", ")));
        sb.append(")");
        return sb.toString();
    }

    private static String getConstructorSignature(ConstructorDeclaration ctor) {
        StringBuilder sb = new StringBuilder();
        if (ctor.isPublic()) sb.append("public ");
        if (ctor.isProtected()) sb.append("protected ");
        sb.append(ctor.getNameAsString()).append("(");
        sb.append(ctor.getParameters().stream().map(Object::toString).collect(Collectors.joining(", ")));
        sb.append(")");
        return sb.toString();
    }
}