package javabridge;

// javabridge/JavaParserBridge.java
import com.github.javaparser.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.*;

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
                cu.findAll(ClassOrInterfaceDeclaration.class).forEach(cls -> {
                    System.out.println(getClassSignature(cls) + " {");
                    // Fields
                    for (FieldDeclaration field : cls.getFields()) {
                        System.out.println("    " + field.toString().replace("\n", " ").trim());
                    }
                    // Only requested methods
                    for (MethodDeclaration method : cls.getMethods()) {
                        if (targetMethods.contains(method.getNameAsString()))
                            System.out.println("    " + method.toString().replace("\n", "\n    ").trim());
                    }
                    System.out.println("}");
                });
                break;
            default:
                System.err.println("Unknown command: " + command);
                System.exit(1);
        }
    }

    private static void printTypeSignature(TypeDeclaration<?> type) {
        if (type instanceof ClassOrInterfaceDeclaration) {
            System.out.println(getClassSignature((ClassOrInterfaceDeclaration) type) + " {");
            // Fields
            for (FieldDeclaration field : ((ClassOrInterfaceDeclaration) type).getFields()) {
                if (field.isPublic() || field.isProtected())
                    System.out.println("    " + field.toString().replace("\n", " ").trim());
            }
            // Constructors
            for (ConstructorDeclaration ctor : ((ClassOrInterfaceDeclaration) type).getConstructors()) {
                if (ctor.isPublic() || ctor.isProtected())
                    System.out.println("    " + getConstructorSignature(ctor) + ";");
            }
            // Methods
            for (MethodDeclaration method : ((ClassOrInterfaceDeclaration) type).getMethods()) {
                if (method.isPublic() || method.isProtected())
                    System.out.println("    " + getMethodSignature(method) + ";");
            }
            System.out.println("}");
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