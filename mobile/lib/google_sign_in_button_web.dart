import 'package:flutter/widgets.dart';
import 'package:google_sign_in_web/web_only.dart' as google_web;

Widget buildGoogleSignInButton({
  required Future<void> Function() onPressed,
}) {
  return Center(child: google_web.renderButton());
}
