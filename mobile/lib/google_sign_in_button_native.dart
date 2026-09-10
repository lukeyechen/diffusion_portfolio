import 'package:flutter/material.dart';

Widget buildGoogleSignInButton({
  required Future<void> Function() onPressed,
}) {
  return SizedBox(
    width: double.infinity,
    child: FilledButton.icon(
      onPressed: onPressed,
      icon: const Icon(Icons.login),
      label: const Text('Sign in with Google'),
    ),
  );
}
