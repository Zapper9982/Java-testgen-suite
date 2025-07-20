#!/bin/bash

set -e

# Paths
JAVABRIDGE_DIR="javabridge"
JAVAPARSER_JAR="lib/javaparser-core-3.25.4.jar"
JAVA_FILE="$JAVABRIDGE_DIR/JavaParserBridge.java"
CLASS_FILE="$JAVABRIDGE_DIR/JavaParserBridge.class"

echo "Cleaning up old class files..."
find "$JAVABRIDGE_DIR" -name 'JavaParserBridge.class' -delete

echo "Compiling JavaParserBridge.java..."
javac -cp "$JAVAPARSER_JAR" "$JAVA_FILE"

if [ ! -f "$CLASS_FILE" ]; then
    echo "ERROR: Compilation failed, $CLASS_FILE not found."
    exit 1
fi

echo "Testing minimal run..."
java -cp "$JAVAPARSER_JAR:$JAVABRIDGE_DIR" javabridge.JavaParserBridge || true

echo "If you want to test with a real Java file, provide the path as an argument:"
echo "  ./javabridge_build_and_test.sh /path/to/SomeClass.java"
if [ -n "$1" ]; then
    echo "Running extract_methods on $1"
    java -cp "$JAVAPARSER_JAR:$JAVABRIDGE_DIR" javabridge.JavaParserBridge extract_methods "$1"
fi

echo "Done."